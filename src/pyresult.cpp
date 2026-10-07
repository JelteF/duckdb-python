#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"

#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_format.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"

using namespace nanobind::literals;

namespace duckdb {

DuckDBPyResult::DuckDBPyResult(unique_ptr<QueryResult> completed) : result(std::move(completed)) {
	if (!result) {
		throw InternalException("PyResult created without a result object");
	}
}

DuckDBPyResult::DuckDBPyResult(unique_ptr<QueryResult> submitted, bool stream_result) {
	if (!submitted) {
		throw InternalException("PyResult created without a result object");
	}
	if (submitted->HasError()) {
		submitted->ThrowError();
	}
	// A statement the planner settles on retained cannot be drained, and neither can a result a
	// delegating collector already built, which carries no buffer.
	const bool can_stream =
	    submitted->HasBufferedData() && submitted->GetStatementProperties().result_eagerness != ResultEagerness::FORCED;
	if (stream_result && can_stream) {
		this->submitted = std::move(submitted);
		return;
	}
	DuckDBPyConnection::CompleteQuery(*submitted);
	result = std::move(submitted);
}

void DuckDBPyResult::EnsureStream() {
	if (submitted) {
		stream = make_uniq<QueryResultStream<ChunkFormat>>(std::move(submitted));
	}
}

const vector<Identifier> &DuckDBPyResult::ResultNames() const {
	if (!names_override.empty()) {
		return names_override;
	}
	if (stream) {
		return stream->GetNames();
	}
	return submitted ? submitted->GetNames() : result->GetNames();
}

void DuckDBPyResult::CloseStream() {
	if (!stream && !submitted) {
		return;
	}
	auto close = [&]() {
		if (stream) {
			stream->Close();
		}
		if (submitted) {
			submitted->Close();
		}
	};
	// Ending the query waits for its running tasks, and a task inside a Python UDF cannot finish
	// until the GIL is free.
	if (duckdb::PyUtil::GilCheck()) {
		nb::gil_scoped_release release;
		close();
	} else {
		close();
	}
}

DuckDBPyResult::~DuckDBPyResult() {
	// The destructor must run with the GIL held: `result` and `current_chunk`
	// can transitively own Python references (registered
	// objects, arrow release callbacks, PYTHON_OBJECT vector values, etc.),
	// whose teardown calls into the Python C API. Releasing the GIL here
	// (as the previous implementation did) causes Py_DECREF / PyObject_Free
	// to run without a valid PyThreadState — see duckdb-python#456.
	try {
		D_ASSERT(duckdb::PyUtil::GilCheck());
		CloseStream();
		stream.reset();
		submitted.reset();
		result.reset();
		current_chunk.reset();
	} catch (...) { // NOLINT
	}
}

const ClientProperties &DuckDBPyResult::GetClientProperties() const {
	if (stream) {
		return stream->GetClientProperties();
	}
	return submitted ? submitted->client_properties : result->client_properties;
}

vector<string> DuckDBPyResult::GetNames() {
	if (Empty()) {
		throw InternalException("Calling GetNames without a result object");
	}
	return IdentifiersToStrings(ResultNames());
}

const vector<LogicalType> &DuckDBPyResult::GetTypes() const {
	if (Empty()) {
		throw InternalException("Calling GetTypes without a result object");
	}
	if (stream) {
		return stream->GetTypes();
	}
	return submitted ? submitted->GetTypes() : result->GetTypes();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchChunk() {
	if (Empty()) {
		throw InternalException("FetchChunk called without a result object");
	}
	return FetchNext();
}

unique_ptr<DataChunk> DuckDBPyResult::FetchStreamChunk() {
	while (true) {
		unique_ptr<DataChunk> chunk;
		auto state = stream->TryFetch(chunk);
		if (chunk) {
			return chunk;
		}
		if (state == QueryResultState::FINISHED) {
			return nullptr;
		}
		if (state == QueryResultState::EXECUTION_ERROR) {
			stream->GetErrorObject().Throw();
		}
		{
			nb::gil_scoped_acquire gil;
			if (PyErr_CheckSignals() != 0) {
				throw std::runtime_error("Query interrupted");
			}
		}
		state = stream->ExecuteTask();
		if (state == QueryResultState::BLOCKED || state == QueryResultState::NO_TASKS_AVAILABLE) {
			stream->WaitForTask();
		}
	}
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNext() {
	EnsureStream();
	if (stream) {
		auto chunk = FetchStreamChunk();
		if (chunk) {
			chunk->Flatten();
		}
		return chunk;
	}
	auto chunk = result->Fetch();
	if (result->HasError()) {
		result->ThrowError();
	}
	return chunk;
}

unique_ptr<DataChunk> DuckDBPyResult::FetchNextRaw() {
	EnsureStream();
	if (stream) {
		return FetchStreamChunk();
	}
	auto chunk = result->FetchRaw();
	if (result->HasError()) {
		result->ThrowError();
	}
	return chunk;
}

unique_ptr<DataChunk> DuckDBPyResult::TakeBufferedRows() {
	unique_ptr<DataChunk> remainder;
	if (current_chunk && chunk_offset < current_chunk->size() && !StreamEnded()) {
		remainder = make_uniq<DataChunk>();
		remainder->Initialize(Allocator::DefaultAllocator(), current_chunk->GetTypes());
		current_chunk->Copy(*remainder, chunk_offset);
	}
	current_chunk.reset();
	chunk_offset = 0;
	return remainder;
}

void DuckDBPyResult::Retain() {
	if (submitted) {
		{
			D_ASSERT(duckdb::PyUtil::GilCheck());
			nb::gil_scoped_release release;
			DuckDBPyConnection::CompleteQuery(*submitted);
		}
		result = std::move(submitted);
		return;
	}
	if (!stream) {
		return;
	}
	auto collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), stream->GetTypes());
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		if (auto buffered = TakeBufferedRows()) {
			collection->Append(*buffered);
		}
		while (auto chunk = FetchStreamChunk()) {
			collection->Append(*chunk);
		}
	}
	auto retained = make_uniq<QueryResult>(stream->GetStatementType(), stream->GetStatementProperties(),
	                                       stream->GetNames(), std::move(collection), stream->GetClientProperties());
	CloseStream();
	stream.reset();
	result = std::move(retained);
}

Optional<nb::tuple> DuckDBPyResult::Fetchone() {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}
	{
		nb::gil_scoped_release release;
		if (!current_chunk || chunk_offset >= current_chunk->size() || StreamEnded()) {
			current_chunk = FetchNext();
			chunk_offset = 0;
		}
	}

	if (!current_chunk || current_chunk->size() == 0) {
		return nb::none();
	}
	auto &types = GetTypes();
	auto &client_properties = GetClientProperties();
	duckdb::PyUtil::TupleBuilder row(types.size());
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto &mask = FlatVector::Validity(current_chunk->data[col_idx]);
		if (!mask.RowIsValid(chunk_offset)) {
			row.append(nb::none());
		} else {
			auto val = current_chunk->data[col_idx].GetValue(chunk_offset);
			row.append(PythonObject::FromValue(val, types[col_idx], client_properties));
		}
	}
	chunk_offset++;
	return row.take();
}

nb::list DuckDBPyResult::Fetchmany(idx_t size) {
	nb::list res;
	for (idx_t i = 0; i < size; i++) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

nb::list DuckDBPyResult::Fetchall() {
	nb::list res;
	while (true) {
		auto fres = Fetchone();
		if (fres.is_none()) {
			break;
		}
		res.append(fres);
	}
	return res;
}

nb::dict DuckDBPyResult::FetchNumpy() {
	return FetchNumpyInternal();
}

void DuckDBPyResult::FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name) {
	if (GetTypes()[col_idx].id() == LogicalTypeId::ENUM) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		auto pandas_categorical = import_cache.pandas.Categorical();
		auto categorical_dtype = import_cache.pandas.CategoricalDtype();
		if (!pandas_categorical || !categorical_dtype) {
			throw InvalidInputException("'pandas' is required for this operation but it was not installed");
		}

		// first we (might) need to create the categorical type
		if (categories_type.find(col_idx) == categories_type.end()) {
			// Equivalent to: pandas.CategoricalDtype(['a', 'b'], ordered=True)
			categories_type[col_idx] = categorical_dtype(categories[col_idx], true);
		}
		// Equivalent to: pandas.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
		res[name] = pandas_categorical.attr("from_codes")(conversion.ToArray(col_idx),
		                                                  nb::arg("dtype") = categories_type[col_idx]);
		if (!conversion.ToPandas()) {
			res[name] = res[name].attr("to_numpy")();
		}
	} else {
		res[name] = conversion.ToArray(col_idx);
	}
}

void InsertCategory(const vector<LogicalType> &types, unordered_map<idx_t, nb::list> &categories) {
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto &type = types[col_idx];
		if (type.id() == LogicalTypeId::ENUM) {
			// It's an ENUM type, in addition to converting the codes we must convert the categories
			if (categories.find(col_idx) == categories.end()) {
				auto &categories_list = EnumType::GetValuesInsertOrder(type);
				auto categories_size = EnumType::GetSize(type);
				for (idx_t i = 0; i < categories_size; i++) {
					categories[col_idx].append(nb::cast(categories_list.GetValue(i).ToString()));
				}
			}
		}
	}
}

std::unique_ptr<NumpyResultConversion> DuckDBPyResult::InitializeNumpyConversion(bool pandas) {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}

	idx_t initial_capacity = STANDARD_VECTOR_SIZE * 2ULL;
	if (result && result->Format().Is<ChunkFormat>()) {
		initial_capacity = result->RowCount();
	}

	auto conversion =
	    std::make_unique<NumpyResultConversion>(GetTypes(), initial_capacity, GetClientProperties(), pandas);
	return conversion;
}

nb::dict DuckDBPyResult::FetchNumpyInternal(bool chunked, idx_t vectors_per_chunk,
                                            std::unique_ptr<NumpyResultConversion> conversion_p) {
	if (Empty()) {
		throw InvalidInputException("result closed");
	}
	if (!conversion_p) {
		conversion_p = InitializeNumpyConversion();
	}
	auto &conversion = *conversion_p;
	if (!chunked) {
		vectors_per_chunk = NumericLimits<idx_t>::Maximum();
	}

	idx_t count_vec = 0;
	if (vectors_per_chunk > 0) {
		unique_ptr<DataChunk> buffered;
		{
			D_ASSERT(duckdb::PyUtil::GilCheck());
			nb::gil_scoped_release release;
			buffered = TakeBufferedRows();
		}
		if (buffered) {
			conversion.Append(*buffered);
			count_vec++;
		}
	}
	for (; count_vec < vectors_per_chunk; count_vec++) {
		unique_ptr<DataChunk> chunk;
		{
			D_ASSERT(duckdb::PyUtil::GilCheck());
			nb::gil_scoped_release release;
			chunk = FetchNextRaw();
		}
		if (!chunk || chunk->size() == 0) {
			break;
		}
		conversion.Append(*chunk);
	}
	InsertCategory(GetTypes(), categories);

	// now that we have materialized the result in contiguous arrays, construct the actual NumPy arrays or categorical
	// types
	nb::dict res;
	auto names = ResultNames();
	QueryResult::DeduplicateColumns(names);
	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		auto &name = names[col_idx];
		FillNumpy(res, col_idx, conversion, name.c_str());
	}
	return res;
}

static void ReplaceDFColumn(PandasDataFrame &df, const char *col_name, idx_t idx, const nb::handle &new_value) {
	df.attr("drop")("columns"_a = col_name, "inplace"_a = true);
	df.attr("insert")(idx, col_name, new_value, "allow_duplicates"_a = false);
}

// TODO: unify these with an enum/flag to indicate which conversions to do
void DuckDBPyResult::ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const {
	auto names = nb::cast<vector<string>>(df.attr("columns"));

	auto &types = GetTypes();
	for (idx_t i = 0; i < types.size(); i++) {
		if (types[i] == LogicalType::TIMESTAMP_TZ) {
			// first localize to UTC then convert to timezone_config
			auto utc_local = df[names[i].c_str()].attr("dt").attr("tz_localize")("UTC");
			auto new_value = utc_local.attr("dt").attr("tz_convert")(GetClientProperties().time_zone);
			// We need to create the column anew because the exact dt changed to a new timezone
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		} else if (date_as_object && types[i] == LogicalType::DATE) {
			nb::object new_value = df[names[i].c_str()].attr("dt").attr("date");
			ReplaceDFColumn(df, names[i].c_str(), i, new_value);
		}
	}
}

static nb::object ConvertNumpyDtype(nb::handle numpy_array) {
	D_ASSERT(duckdb::PyUtil::GilCheck());
	auto &import_cache = *DuckDBPyConnection::ImportCache();

	auto dtype = numpy_array.attr("dtype");
	if (!duckdb::PyUtil::IsInstance(numpy_array, import_cache.numpy.ma.masked_array())) {
		return dtype;
	}

	auto numpy_type = ConvertNumpyType(dtype);
	switch (numpy_type.type) {
	case NumpyNullableType::BOOL: {
		return import_cache.pandas.BooleanDtype()();
	}
	case NumpyNullableType::UINT_8: {
		return import_cache.pandas.UInt8Dtype()();
	}
	case NumpyNullableType::UINT_16: {
		return import_cache.pandas.UInt16Dtype()();
	}
	case NumpyNullableType::UINT_32: {
		return import_cache.pandas.UInt32Dtype()();
	}
	case NumpyNullableType::UINT_64: {
		return import_cache.pandas.UInt64Dtype()();
	}
	case NumpyNullableType::INT_8: {
		return import_cache.pandas.Int8Dtype()();
	}
	case NumpyNullableType::INT_16: {
		return import_cache.pandas.Int16Dtype()();
	}
	case NumpyNullableType::INT_32: {
		return import_cache.pandas.Int32Dtype()();
	}
	case NumpyNullableType::INT_64: {
		return import_cache.pandas.Int64Dtype()();
	}
	case NumpyNullableType::FLOAT_32:
	case NumpyNullableType::FLOAT_64:
	case NumpyNullableType::FLOAT_16: // there is no pandas.Float16Dtype
	default:
		return dtype;
	}
}

PandasDataFrame DuckDBPyResult::FrameFromNumpy(bool date_as_object, const nb::handle &o) {
	D_ASSERT(duckdb::PyUtil::GilCheck());
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas) {
		throw InvalidInputException("'pandas' is required for this operation but it was not installed");
	}

	nb::object items = o.attr("items")();
	for (const nb::handle &item : items) {
		// Each item is a tuple of (key, value)
		auto key_value = nb::cast<nb::tuple>(item);
		nb::handle key = key_value[0];   // Access the first element (key)
		nb::handle value = key_value[1]; // Access the second element (value)

		auto dtype = ConvertNumpyDtype(value);
		if (duckdb::PyUtil::IsInstance(value, import_cache.numpy.ma.masked_array())) {
			// o[key] = pd.Series(value.filled(pd.NA), dtype=dtype)
			auto series = pandas.attr("Series")(value.attr("data"), nb::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			o.attr("__setitem__")(key, series);
		}
	}

	PandasDataFrame df = nb::cast<PandasDataFrame>(pandas.attr("DataFrame").attr("from_dict")(o));
	// Convert TZ and (optionally) Date types
	ConvertDateTimeTypes(df, date_as_object);

	auto names = nb::cast<vector<string>>(df.attr("columns"));
	D_ASSERT(GetTypes().size() == names.size());
	return df;
}

PandasDataFrame DuckDBPyResult::FetchDF(bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(false, 1, std::move(conversion)));
}

PandasDataFrame DuckDBPyResult::FetchDFChunk(idx_t num_of_vectors, bool date_as_object) {
	auto conversion = InitializeNumpyConversion(true);
	return FrameFromNumpy(date_as_object, FetchNumpyInternal(true, num_of_vectors, std::move(conversion)));
}

nb::dict DuckDBPyResult::FetchPyTorch() {
	auto result_dict = FetchNumpyInternal();
	auto from_numpy = nb::module_::import_("torch").attr("from_numpy");
	for (auto item : result_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
		result_dict[item.first] = from_numpy(item.second);
	}
	return result_dict;
}

nb::dict DuckDBPyResult::FetchTF() {
	auto result_dict = FetchNumpyInternal();
	auto convert_to_tensor = nb::module_::import_("tensorflow").attr("convert_to_tensor");
	for (auto item : result_dict) { // nanobind dict iteration yields std::pair<handle,handle> by value
		result_dict[item.first] = convert_to_tensor(item.second);
	}
	return result_dict;
}

// A SelectStatement over a ColumnDataRef rather than a relation: a RelationStatement would
// stringify the whole collection when the query is submitted.
static unique_ptr<SelectStatement> MakeColumnDataScanStatement(unique_ptr<ColumnDataCollection> collection,
                                                               const vector<Identifier> &names) {
	// The binder rejects duplicate column names; callers restore the originals afterwards.
	auto deduplicated_names = names;
	QueryResult::DeduplicateColumns(deduplicated_names);
	auto table_ref = make_uniq<ColumnDataRef>(std::move(collection), std::move(deduplicated_names));
	table_ref->alias = "materialized"; // binding asserts on an unset alias
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->from_table = std::move(table_ref);
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(select_node);
	return select;
}

void DuckDBPyResult::PromoteMaterializedToArrow(idx_t batch_size) {
	D_ASSERT(result->Format().Is<ChunkFormat>());
	auto client_context = result->client_properties.client_context;
	if (!client_context) {
		throw InternalException("Cannot promote result to Arrow: the originating client context is gone");
	}
	auto context = client_context->shared_from_this();
	auto names = ResultNames();
	auto select = MakeColumnDataScanStatement(result->TakeCollection(), names);

	unique_ptr<QueryResult> new_result;
	{
		D_ASSERT(duckdb::PyUtil::GilCheck());
		nb::gil_scoped_release release;
		new_result = context->Submit(std::move(select), QueryParameters(make_shared_ptr<ArrowFormat>(batch_size)));
		DuckDBPyConnection::CompleteQuery(*new_result);
	}
	names_override = std::move(names); // restore names de-duplicated by re-binding
	result = std::move(new_result);
}

template <typename T>
T DuckDBPyResult::RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names) {
	D_ASSERT(!Empty());
	auto client_properties = GetClientProperties();
	if (!client_properties.client_context) {
		throw ConnectionException("Cannot fetch arrow schema without a valid connection");
	}
	auto ctx = client_properties.client_context->shared_from_this();

	auto identifiers = ResultNames();
	if (dedup_col_names) {
		QueryResult::DeduplicateColumns(identifiers);
	}
	auto names = IdentifiersToStrings(identifiers);

	ArrowSchema arrow_schema;
	ctx->RunFunctionInTransaction(
	    [&] { ArrowConverter::ToArrowSchema(&arrow_schema, GetTypes(), names, client_properties); });

	return fun(arrow_schema);
}

duckdb::pyarrow::Table DuckDBPyResult::MaterializedResultToArrowTable(const ArrowSchema &arrow_schema,
                                                                      const idx_t rows_per_batch) {
	Retain();
	D_ASSERT(result);
	D_ASSERT(result->Format().Is<ChunkFormat>() || result->Format().Is<ArrowFormat>());

	auto pyarrow_schema = pyarrow::ToPyArrowSchema(arrow_schema);
	if (result->Format().Is<ChunkFormat>()) {
		PromoteMaterializedToArrow(rows_per_batch);
	}
	nb::list batches;
	auto arrays = result->TakeCollection<ArrowFormat>();
	for (auto &array : *arrays) {
		// The collection's arrays are shared and immutable, so pyarrow gets an export that keeps its
		// array alive instead of the array itself
		ArrowArray data;
		ArrowFormat::ShareArray(array)->MoveTo(data);
		TransformDuckToArrowChunk(pyarrow_schema, data, batches);
	}
	return pyarrow::ToArrowTable(std::move(batches), pyarrow_schema);
}

duckdb::pyarrow::Table DuckDBPyResult::FetchArrowTable(const idx_t rows_per_batch, const bool to_polars) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}

	return RunWithArrowSchema<duckdb::pyarrow::Table>(
	    [&](const ArrowSchema &schema) -> duckdb::pyarrow::Table {
		    return MaterializedResultToArrowTable(schema, rows_per_batch);
	    },
	    to_polars);
}

static void CheckBatchSize(idx_t rows_per_batch) {
	if (rows_per_batch == 0) {
		throw std::runtime_error("Approximate Batch Size of Record Batch MUST be higher than 0");
	}
}

namespace {

//! pyarrow releases an imported stream with the GIL held, and releasing the engine's stream ends
//! the query and joins its tasks, one of which may be inside a Python UDF waiting for the GIL.
//! Pulling batches has the same shape when a caller holds the GIL, so every callback drops it.
template <class FUN>
auto WithoutGil(FUN &&fun) -> decltype(fun()) {
	if (duckdb::PyUtil::GilCheck()) {
		nb::gil_scoped_release release;
		return fun();
	}
	return fun();
}

//! The engine's stream and, for a result whose query already ended, the context its batches are
//! converted under. An open stream keeps its own context; a retained result released it.
struct EngineStreamHandoff {
	ArrowArrayStream engine;
	shared_ptr<const ClientContext> context;
};

ArrowArrayStream &EngineStream(ArrowArrayStream *stream) {
	return static_cast<EngineStreamHandoff *>(stream->private_data)->engine;
}

int ForwardGetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
	auto &engine = EngineStream(stream);
	return WithoutGil([&]() { return engine.get_schema(&engine, out); });
}

int ForwardGetNext(ArrowArrayStream *stream, ArrowArray *out) {
	auto &engine = EngineStream(stream);
	return WithoutGil([&]() { return engine.get_next(&engine, out); });
}

const char *ForwardGetLastError(ArrowArrayStream *stream) {
	auto &engine = EngineStream(stream);
	return engine.get_last_error(&engine);
}

void ForwardRelease(ArrowArrayStream *stream) {
	if (!stream->release) {
		return;
	}
	auto handoff = static_cast<EngineStreamHandoff *>(stream->private_data);
	WithoutGil([&]() {
		if (handoff->engine.release) {
			handoff->engine.release(&handoff->engine);
		}
		delete handoff;
	});
	stream->private_data = nullptr;
	stream->release = nullptr;
}

//! Converts a chunk result into Arrow batches on the consumer thread. A query's format is fixed when it
//! is submitted, and these were submitted before anyone asked for Arrow, so the engine's ArrowFormat
//! cannot build the batches. ArrowFormat's state is not reused for the conversion either, because it
//! requires a live client context, which a retained result may not have anymore.
class ChunkResultArrowStream {
public:
	ChunkResultArrowStream(unique_ptr<QueryResult> handle, bool drain, idx_t batch_size_p)
	    : client_properties(handle->client_properties), batch_size(batch_size_p) {
		if (drain) {
			chunk_stream = make_uniq<QueryResultStream<ChunkFormat>>(std::move(handle));
		} else {
			result = std::move(handle);
		}
		names = IdentifiersToStrings(chunk_stream ? chunk_stream->GetNames() : result->GetNames());
		if (client_properties.client_context) {
			extension_types = ArrowTypeExtensionData::GetExtensionTypes(*client_properties.client_context, Types());
		}
		stream.private_data = this;
		stream.get_schema = GetSchema;
		stream.get_next = GetNext;
		stream.get_last_error = GetLastError;
		stream.release = Release;
	}

	ArrowArrayStream stream;

private:
	const vector<LogicalType> &Types() const {
		return chunk_stream ? chunk_stream->GetTypes() : result->GetTypes();
	}

	//! False on an error, which is recorded in last_error. A null chunk means the result is exhausted
	bool NextChunk() {
		current_offset = 0;
		if (!chunk_stream) {
			current_chunk = result->Fetch();
			return true;
		}
		// A stream ended by another statement has not been asked yet; Poll records that as its error
		if (chunk_stream->Poll() == QueryResultState::EXECUTION_ERROR) {
			last_error = chunk_stream->GetErrorObject();
			return false;
		}
		if (!chunk_stream->IsOpen()) {
			// The ended stream released its context, which converting a batch would need
			current_chunk.reset();
			return true;
		}
		current_chunk = chunk_stream->Fetch();
		if (!current_chunk && chunk_stream->HasError()) {
			last_error = chunk_stream->GetErrorObject();
			return false;
		}
		return true;
	}

	int FetchBatch(ArrowArray &out) {
		unique_ptr<ArrowAppender> appender;
		idx_t count = 0;
		while (count < batch_size) {
			if (!current_chunk || current_offset >= current_chunk->size()) {
				if (exhausted) {
					break;
				}
				if (!NextChunk()) {
					exhausted = true;
					return -1;
				}
				if (!current_chunk) {
					exhausted = true;
					break;
				}
				continue;
			}
			if (!appender) {
				appender = make_uniq<ArrowAppender>(Types(), batch_size, client_properties, extension_types);
			}
			auto to_append = MinValue(batch_size - count, current_chunk->size() - current_offset);
			appender->Append(*current_chunk, current_offset, current_offset + to_append, current_chunk->size());
			current_offset += to_append;
			count += to_append;
		}
		if (count > 0) {
			out = appender->Finalize();
		}
		return 0;
	}

	static ChunkResultArrowStream &Get(ArrowArrayStream *stream) {
		return *static_cast<ChunkResultArrowStream *>(stream->private_data);
	}

	static int GetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
		if (!stream->release) {
			return -1;
		}
		out->release = nullptr;
		auto &self = Get(stream);
		if (self.chunk_stream && self.chunk_stream->HasError()) {
			self.last_error = self.chunk_stream->GetErrorObject();
			return -1;
		}
		try {
			ArrowConverter::ToArrowSchema(out, self.Types(), self.names, self.client_properties);
		} catch (std::exception &ex) {
			self.last_error = ErrorData(ex);
			return -1;
		}
		return 0;
	}

	static int GetNext(ArrowArrayStream *stream, ArrowArray *out) {
		if (!stream->release) {
			return -1;
		}
		out->release = nullptr;
		auto &self = Get(stream);
		try {
			return self.FetchBatch(*out);
		} catch (std::exception &ex) {
			self.last_error = ErrorData(ex);
			return -1;
		}
	}

	static const char *GetLastError(ArrowArrayStream *stream) {
		if (!stream->release) {
			return "stream was released";
		}
		return Get(stream).last_error.Message().c_str();
	}

	static void Release(ArrowArrayStream *stream) {
		if (!stream->release) {
			return;
		}
		stream->release = nullptr;
		delete &Get(stream);
	}

private:
	ClientProperties client_properties;
	idx_t batch_size;
	//! Exactly one of these is set: the stream when the handle could still be drained, else the retained result
	unique_ptr<QueryResultStream<ChunkFormat>> chunk_stream;
	unique_ptr<QueryResult> result;
	vector<string> names;
	unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>> extension_types;
	unique_ptr<DataChunk> current_chunk;
	idx_t current_offset = 0;
	bool exhausted = false;
	ErrorData last_error;
};

//! Releases a stream that was never handed over, for example when the import into pyarrow throws
struct ArrowArrayStreamGuard {
	ArrowArrayStream stream;
	~ArrowArrayStreamGuard() {
		if (stream.release) {
			stream.release(&stream);
		}
	}
};

} // namespace

ArrowArrayStream DuckDBPyResult::FetchArrowArrayStream(idx_t rows_per_batch) {
	if (stream) {
		Retain();
	}
	auto &client_context = GetClientProperties().client_context;
	auto context = client_context ? client_context->shared_from_this() : shared_ptr<const ClientContext>();
	const bool drain = submitted != nullptr;
	auto handle = drain ? std::move(submitted) : std::move(result);
	current_chunk.reset();
	chunk_offset = 0;
	const auto result_stream = new ChunkResultArrowStream(std::move(handle), drain, rows_per_batch);
	auto handoff = new EngineStreamHandoff {result_stream->stream, std::move(context)};
	ArrowArrayStream forwarding;
	forwarding.get_schema = ForwardGetSchema;
	forwarding.get_next = ForwardGetNext;
	forwarding.get_last_error = ForwardGetLastError;
	forwarding.release = ForwardRelease;
	forwarding.private_data = handoff;
	return forwarding;
}

//! An Arrow result was converted already and has no collection the engine's stream could read
bool DuckDBPyResult::IsArrow() const {
	return result && result->Format().Is<ArrowFormat>();
}

duckdb::pyarrow::RecordBatchReader DuckDBPyResult::FetchRecordBatchReader(idx_t rows_per_batch) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}
	CheckBatchSize(rows_per_batch);

	if (IsArrow()) {
		constexpr bool dedup_column_names = false;
		auto reader = RunWithArrowSchema<duckdb::pyarrow::RecordBatchReader>(
		    [&](const ArrowSchema &schema) -> duckdb::pyarrow::RecordBatchReader {
			    const auto table = MaterializedResultToArrowTable(schema, rows_per_batch);
			    return nb::cast<duckdb::pyarrow::RecordBatchReader>(
			        table.attr("to_reader")(nb::arg("max_chunksize") = rows_per_batch));
		    },
		    dedup_column_names);
		result.reset();
		return reader;
	}
	auto pyarrow_lib_module = nb::module_::import_("pyarrow").attr("lib");
	auto record_batch_reader_func = pyarrow_lib_module.attr("RecordBatchReader").attr("_import_from_c");
	ArrowArrayStreamGuard guard {FetchArrowArrayStream(rows_per_batch)};
	nb::object record_batch_reader = record_batch_reader_func((uint64_t)&guard.stream); // NOLINT
	return nb::cast<duckdb::pyarrow::RecordBatchReader>(record_batch_reader);
}

static void ArrowArrayStreamPyCapsuleDestructor(void *data) noexcept {
	if (!data) {
		return;
	}
	auto arrow_stream = reinterpret_cast<ArrowArrayStream *>(data);
	if (arrow_stream->release) {
		arrow_stream->release(arrow_stream);
	}
	delete arrow_stream;
}

nb::object DuckDBPyResult::FetchArrowCapsule(const idx_t rows_per_batch) {
	if (Empty()) {
		throw InvalidInputException("There is no query result");
	}
	CheckBatchSize(rows_per_batch);

	if (IsArrow()) {
		constexpr bool dedup_column_names = false;
		auto capsule = RunWithArrowSchema<nb::object>(
		    [&](const ArrowSchema &schema) -> nb::object {
			    const auto table = MaterializedResultToArrowTable(schema, rows_per_batch);
			    return table.attr("__arrow_c_stream__")();
		    },
		    dedup_column_names);
		result.reset();
		return capsule;
	}
	auto inner_stream = FetchArrowArrayStream(rows_per_batch);
	auto arrow_stream = new ArrowArrayStream();
	*arrow_stream = inner_stream;
	return nb::capsule(arrow_stream, "arrow_array_stream", ArrowArrayStreamPyCapsuleDestructor);
}

nb::list DuckDBPyResult::GetDescription(const vector<string> &names, const vector<LogicalType> &types) {
	nb::list desc;

	for (idx_t col_idx = 0; col_idx < names.size(); col_idx++) {
		auto py_name = nb::str(names[col_idx].c_str(), names[col_idx].size());
		auto py_type = DuckDBPyType(types[col_idx]);
		desc.append(nb::make_tuple(py_name, py_type, nb::none(), nb::none(), nb::none(), nb::none(), nb::none()));
	}
	return desc;
}

void DuckDBPyResult::Close() {
	CloseStream();
	stream.reset();
	submitted.reset();
	result.reset();
	current_chunk.reset();
	chunk_offset = 0;
}

} // namespace duckdb

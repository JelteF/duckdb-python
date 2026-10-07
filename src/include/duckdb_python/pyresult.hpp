//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pyresult.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/numpy/numpy_result_conversion.hpp"
#include "duckdb.hpp"
#include "duckdb/main/query_result_stream.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/dataframe.hpp"

namespace duckdb {

struct DuckDBPyResult {
public:
	//! A result that has run to completion: it holds its rows, or is an Arrow result
	explicit DuckDBPyResult(unique_ptr<QueryResult> completed);
	//! A freshly submitted handle. Opened as a stream when the caller asks for one and the statement
	//! can be drained, otherwise run to completion. Call with the GIL released.
	DuckDBPyResult(unique_ptr<QueryResult> submitted, bool stream_result);
	~DuckDBPyResult();

public:
	Optional<nb::tuple> Fetchone();

	nb::list Fetchmany(idx_t size);

	nb::list Fetchall();

	nb::dict FetchNumpy();

	nb::dict FetchNumpyInternal(bool chunked = false, idx_t vectors_per_chunk = 1,
	                            std::unique_ptr<NumpyResultConversion> conversion = nullptr);

	PandasDataFrame FetchDF(bool date_as_object);

	PandasDataFrame FetchDFChunk(const idx_t vectors_per_chunk = 1, bool date_as_object = false);

	nb::dict FetchPyTorch();

	nb::dict FetchTF();

	duckdb::pyarrow::Table FetchArrowTable(idx_t rows_per_batch, bool to_polars);
	duckdb::pyarrow::RecordBatchReader FetchRecordBatchReader(idx_t rows_per_batch = 1000000);
	nb::object FetchArrowCapsule(idx_t rows_per_batch = 1000000);

	static nb::list GetDescription(const vector<string> &names, const vector<LogicalType> &types);

	void Close();

	unique_ptr<DataChunk> FetchChunk();

	vector<string> GetNames();
	const vector<LogicalType> &GetTypes() const;

	const ClientProperties &GetClientProperties() const;

private:
	void FillNumpy(nb::dict &res, idx_t col_idx, NumpyResultConversion &conversion, const char *name);

	PandasDataFrame FrameFromNumpy(bool date_as_object, const nb::handle &o);

	void ConvertDateTimeTypes(PandasDataFrame &df, bool date_as_object) const;
	//! The names the Python layer reports, see the definition for why this is not always the result's own.
	const vector<Identifier> &ResultNames() const;
	bool Empty() const {
		return !result && !stream && !submitted;
	}
	//! The Arrow readers take the handle itself, so the stream is not opened before the first
	//! consumer call decides.
	void EnsureStream();
	//! Flat vectors, for the row fetch
	unique_ptr<DataChunk> FetchNext();
	unique_ptr<DataChunk> FetchNextRaw();
	unique_ptr<DataChunk> FetchStreamChunk();
	//! Rows a row fetch popped but has not returned yet. Once the stream's query has ended
	//! underneath them they are dropped, so that the engine reports why on the next fetch.
	unique_ptr<DataChunk> TakeBufferedRows();
	//! Takes the context lock: call with the GIL released, the engine acquires the GIL under it.
	bool StreamEnded() const {
		return stream && !stream->IsOpen();
	}
	void Retain();
	void CloseStream();
	std::unique_ptr<NumpyResultConversion> InitializeNumpyConversion(bool pandas = false);

	//! Re-feed a retained result's collection through a query in the ArrowFormat on the user's own
	//! context, which converts in parallel and yields an Arrow result in its place.
	void PromoteMaterializedToArrow(idx_t batch_size);

	template <typename T>
	T RunWithArrowSchema(const std::function<T(const ArrowSchema &)> &fun, bool dedup_col_names);
	duckdb::pyarrow::Table MaterializedResultToArrowTable(const ArrowSchema &arrow_schema, idx_t rows_per_batch);
	//! The stream's private data owns the handle, and its callbacks run without the GIL.
	ArrowArrayStream FetchArrowArrayStream(idx_t rows_per_batch);
	bool IsArrow() const;

private:
	idx_t chunk_offset = 0;
	//! The completed result, when the rows were retained or converted to Arrow
	unique_ptr<QueryResult> result;
	//! The open stream the rows are drained through
	unique_ptr<QueryResultStream<ChunkFormat>> stream;
	//! A handle that can be drained but has not been consumed yet
	unique_ptr<QueryResult> submitted;
	//! Set only when the result was re-bound (promotion to Arrow de-duplicates column names
	//! and core exposes no setter), so the original names survive. Empty means "use result's".
	vector<Identifier> names_override;
	unique_ptr<DataChunk> current_chunk;
	// Holds the categories of Categorical/ENUM types
	unordered_map<idx_t, nb::list> categories;
	// Holds the categorical type of Categorical/ENUM types
	unordered_map<idx_t, nb::object> categories_type;
};

} // namespace duckdb

import pytest
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.openGauss import OpenGaussStore

PARAMS = {
    "host": "localhost",
    "user": "postgres",
    "password": "mark90",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 2

try:
    import asyncpg
    import psycopg2
    postgres_not_available = False
except ImportError:
    postgres_not_available = True

def _get_sample_vector(num: float) -> List[float]:
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)

# Fixtures
@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2
    return psycopg2.connect(**PARAMS)

@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        c.execute(f"CREATE DATABASE {TEST_DB}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE {TEST_DB}")
        conn.commit()

@pytest.fixture()
def opengauss_store(db: None) -> Generator[OpenGaussStore, None, None]:
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        use_bm25=True,
    )
    yield store
    asyncio.run(store.close())

@pytest.fixture()
def opengauss_store_indexed(db: None) -> Generator[OpenGaussStore, None, None]:
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        indexed_metadata_keys={("test_text", "text"), ("test_int", "int")},
    )
    yield store
    asyncio.run(store.close())

@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            extra_info={"test_num": 1},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="consectetur adipiscing elit",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_list": ["test_value"]},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="sed do eiusmod tempor",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_2": "test_val_2"},
            embedding=_get_sample_vector(0.1),
        ),
    ]

# 测试用例
@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_bm25_hybrid_search(
    opengauss_store: OpenGaussStore
):
    nodes = [
        TextNode(text="quick brown fox", id_="fox"),
        TextNode(text="lazy dog", id_="dog"),
    ]
    await opengauss_store.async_add(nodes)
    
    query = VectorStoreQuery(
        query_str="fox",
        mode=VectorStoreQueryMode.SPARSE,
        sparse_top_k=2
    )
    results = await opengauss_store.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == "fox"

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
def test_metadata_index_creation(
    opengauss_store_indexed: OpenGaussStore
):
    with opengauss_store_indexed._session() as session:
        indexes = session.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = :schema 
            AND tablename LIKE :table
            """, 
            {"schema": TEST_SCHEMA_NAME, "table": f"data_{TEST_TABLE_NAME}"}
        ).fetchall()
        
        index_names = [idx[0] for idx in indexes]
        assert f"{TEST_TABLE_NAME}_idx_test_text_text" in index_names
        assert f"{TEST_TABLE_NAME}_idx_test_int_int" in index_names

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_from_params_initialization():
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name="test_table",
        use_bm25=True
    )
    
    assert store.use_bm25 is True
    assert store.table_name == "test_table"
    await store.close()

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_query_with_score(
    opengauss_store: OpenGaussStore,
    node_embeddings: List[TextNode]
):
    await opengauss_store.async_add(node_embeddings)
    
    results = await opengauss_store._aquery_with_score(
        embedding=_get_sample_vector(0.5),
        limit=2
    )
    
    assert len(results) == 2
    assert isinstance(results[0].similarity, float)
    assert 0 <= results[0].similarity <= 1

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
def test_bm25_query_generation(opengauss_store: OpenGaussStore):
    from sqlalchemy import select
    
    stmt = opengauss_store._build_sparse_query("test query", 10)
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    
    assert "<&>" in str(compiled)
    assert "to_tsquery" in str(compiled)

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_metadata_filtered_query(
    opengauss_store_indexed: OpenGaussStore
):
    nodes = [
        TextNode(text="apple", id_="a1", metadata={"test_int": 1}),
        TextNode(text="banana", id_="b1", metadata={"test_int": 2}),
    ]
    await opengauss_store_indexed.async_add(nodes)
    
    filters = MetadataFilters(filters=[
        MetadataFilter(key="test_int", value=1, operator=FilterOperator.EQ)
    ])
    
    query = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        filters=filters
    )
    
    results = await opengauss_store_indexed.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == "a1"

@pytest.mark.skipif(postgres_not_available, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_connection_handling():
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME
    )
    
    assert store._engine is not None
    await store.aclear()
    await store.close()
    
    with pytest.raises(Exception):
        await store.aquery(VectorStoreQuery(query_embedding=[0.1, 0.1]))

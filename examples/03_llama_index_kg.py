import os
from typing import Literal

from dotenv import load_dotenv
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPGStore
import pandas as pd
from llama_index.core import Document, PropertyGraphIndex
from llama_index.llms.openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

username = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
url = os.environ.get('NEO4J_URI')

graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=url,
)

news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")
text = """
The Taj Mahal (/ˌtɑːdʒ məˈhɑːl, ˌtɑːʒ-/; lit. 'Crown of the Palace') is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by the fifth Mughal emperor, Shah Jahan (r. 1628–1658) to house the tomb of his beloved wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall.

Construction of the mausoleum was completed in 1648, but work continued on other phases of the project for another five years. The first ceremony held at the mausoleum was an observance by Shah Jahan, on 6 February 1643, of the 12th anniversary of the death of Mumtaz Mahal. The Taj Mahal complex is believed to have been completed in its entirety in 1653 at a cost estimated at the time to be around ₹5 million, which in 2023 would be approximately ₹35 billion (US$77.8 million).

"""
documents = [Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()]
# documents = [Document(text=text)]
# for document in documents:
#     print(document)

llm = OpenAI(model="gpt-4o", temperature=0.0)
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

#
entities = Literal["PERSON", "LOCATION", "ORGANIZATION", "PRODUCT", "EVENT"]
relations = Literal[
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION",
]
validation_schema = {
    "Person": ["WORKS_AT", "BOARD_MEMBER", "CEO", "HAS_EVENT"],
    "Organization": [
        "SUPPLIER_OF",
        "COMPETITOR",
        "PARTNERSHIP",
        "ACQUISITION",
        "WORKS_AT",
        "SUBSIDIARY",
        "BOARD_MEMBER",
        "CEO",
        "PROVIDES",
        "HAS_EVENT",
        "IN_LOCATION",
    ],
    "Product": ["PROVIDES"],
    "Event": ["HAS_EVENT", "IN_LOCATION"],
    "Location": ["HAPPENED_AT", "IN_LOCATION"],
}
# best practice to use upper-case
# entities = Literal["PERSON", "LOCATION", "MONUMENT"]
# relations = Literal[
#     "LOVED",
#     "BUILDER_OF",
#     "SITUATED",
#     "BUILT_IN",
#     "BUILT_BY"
# ]
#
# validation_schema = {
#     "Person": ["BUILDER_OF", "LOVED"],
#     "MONUMENT":["BUILT_BY","BUILT_IN"],
#     "Location": ["HAPPENED_AT", "IN_LOCATION"],
# }

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    # if false, allows for values outside of the schema
    # useful for using the schema as a suggestion
    strict=True,
)

NUMBER_OF_ARTICLES = 25

index = PropertyGraphIndex.from_documents(
    documents[:NUMBER_OF_ARTICLES],
    kg_extractors=[kg_extractor],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

graph_store.structured_query("""
CREATE VECTOR INDEX my_entity IF NOT EXISTS
FOR (m:`__Entity__`)
ON m.embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
""")
# # Just for inspection
# similarity_threshold = 0.1
# word_edit_distance = 5
# entity = "my_entity"
# data = graph_store.structured_query("""
# MATCH (e:__Entity__)
# CALL {
#   WITH e
#   CALL db.index.vector.queryNodes($entity, 10, e.embedding)
#   YIELD node, score
#   WITH node, score
#   WHERE score > toFLoat($cutoff)
#       AND (toLower(node.name) CONTAINS toLower(e.name) OR toLower(e.name) CONTAINS toLower(node.name)
#            OR apoc.text.distance(toLower(node.name), toLower(e.name)) < $distance)
#       AND labels(e) = labels(node)
#   WITH node, score
#   ORDER BY node.name
#   RETURN collect(node) AS nodes
# }
# WITH distinct nodes
# WHERE size(nodes) > 1
# WITH collect([n in nodes | n.name]) AS results
# UNWIND range(0, size(results)-1, 1) as index
# WITH results, index, results[index] as result
# WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
#         CASE WHEN index <> index2 AND
#             size(apoc.coll.intersection(acc, results[index2])) > 0
#             THEN apoc.coll.union(acc, results[index2])
#             ELSE acc
#         END
# )) as combinedResult
# WITH distinct(combinedResult) as combinedResult
# // extra filtering
# WITH collect(combinedResult) as allCombinedResults
# UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
# WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
# WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
#     WHERE x <> combinedResultIndex
#     AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
# )
# RETURN combinedResult
# """, param_map={'cutoff': similarity_threshold, 'distance': word_edit_distance,'entity':entity})
#
# for row in data:
#     print(row)
#
# graph_store.structured_query("""
# MATCH (e:__Entity__)
# CALL {
#   WITH e
#   CALL db.index.vector.queryNodes($entity, 10, e.embedding)
#   YIELD node, score
#   WITH node, score
#   WHERE score > toFLoat($cutoff)
#       AND (toLower(node.name) CONTAINS toLower(e.name) OR toLower(e.name) CONTAINS toLower(node.name)
#            OR apoc.text.distance(toLower(node.name), toLower(e.name)) < $distance)
#       AND labels(e) = labels(node)
#   WITH node, score
#   ORDER BY node.name
#   RETURN collect(node) AS nodes
# }
# WITH distinct nodes
# WHERE size(nodes) > 1
# WITH collect([n in nodes | n.name]) AS results
# UNWIND range(0, size(results)-1, 1) as index
# WITH results, index, results[index] as result
# WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
#         CASE WHEN index <> index2 AND
#             size(apoc.coll.intersection(acc, results[index2])) > 0
#             THEN apoc.coll.union(acc, results[index2])
#             ELSE acc
#         END
# )) as combinedResult
# WITH distinct(combinedResult) as combinedResult
# // extra filtering
# WITH collect(combinedResult) as allCombinedResults
# UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
# WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
# WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
#     WHERE x <> combinedResultIndex
#     AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
# )
# CALL {
#   WITH combinedResult
# 	UNWIND combinedResult AS name
# 	MATCH (e:__Entity__ {name:name})
# 	WITH e
# 	ORDER BY size(e.name) DESC // prefer longer names to remain after merging
# 	RETURN collect(e) AS nodes
# }
# CALL apoc.refactor.mergeNodes(nodes, {properties: {
#     `.*`: 'discard'
# }})
# YIELD node
# RETURN count(*)
# """, param_map={'cutoff': similarity_threshold, 'distance': word_edit_distance,'entity':entity})

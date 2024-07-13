import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

username = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
uri = os.environ.get('NEO4J_URI')

# Create a driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))


# # Define a function to execute the query
# def run_query(driver, query):
#     with driver.session() as session:
#         result = session.run(query)
#         return result.values()

# Execute the second part of the query
def run_query_with_params(driver, query, params=None):
    if params is None:
        params = {}
    with driver.session() as session:
        result = session.run(query, params)
        return result.values()

# This is the first step
# Identify all nodes that are similar(uses cosine with 0.8 threshold)
# And also nodes that are 5 Levenshtein distance apart
# The first collect aggregates the similar nodes in a list from "CALL db.index.vector.queryNodes('my_entity', 10, e.embedding)"
# The second collect aggregates the names of nodes within each nodes list into a list of lists called results.
query_part1 = """
MATCH (e:__Entity__)
CALL {
  WITH e
  CALL db.index.vector.queryNodes('my_entity', 10, e.embedding)
  YIELD node, score
    WITH node, score
        WHERE score > toFloat(0.8)
        AND (toLower(node.name) CONTAINS toLower(e.name) OR toLower(e.name) CONTAINS toLower(node.name)
        OR apoc.text.distance(toLower(node.name), toLower(e.name)) < 5)
        AND labels(e) = labels(node)
            WITH node, score
                ORDER BY node.name
  RETURN collect(node) AS nodes
}
WITH distinct nodes
WHERE size(nodes) > 1
WITH collect([n in nodes | n.name]) AS results
RETURN results
"""

# Execute the first part of the query to get results
results_part1 = run_query_with_params(driver, query_part1)

# Print the results from the first part
print("Results from the first part:")
for result in results_part1:
    print(result)

# Flatten the results for the next query
flat_results = [name for sublist in results_part1[0] for name in sublist]

print("flat ", flat_results)

# Define the second part of the Cypher query to process results
# Unwind expands the similar node pair list($results) into individual node pair list.
# reduce function iterates over index2 of the $results list
# acc is the accumulator which starts with result
# for each index2, it checks if result is different from $results[index2]
# and if there is an intersection between result and $results[index2]
# If both conditions are true, it unions result and $results[index2] and stores the result in acc.
# If not, it keeps acc unchanged.
# After the reduce operation, the apoc.coll.sort function sorts the combined result.
# This WITH clause ensures that only distinct combinedResult values are passed to the next step.
# Let us understand with an example
# $results = [["NodeA", "NodeB"], ["NodeB", "NodeC"], ["NodeD", "NodeE"], ["NodeE", "NodeF"], ["NodeG"]]
# UNWIND $results as result
# result = ["NodeA", "NodeB"]
# result = ["NodeB", "NodeC"]
# result = ["NodeD", "NodeE"]
# result = ["NodeE", "NodeF"]
# result = ["NodeG"]
# WITH apoc.coll.sort(reduce(...)) as combinedResult
# For each result, it performs the following operations:
# First iteration
# result = ["NodeA", "NodeB"]
# acc starts as ["NodeA", "NodeB"].
# Loop through $results:
# Index 0: Same as result, skip.
# Index 1: ["NodeB", "NodeC"] intersects with ["NodeA", "NodeB"].
# Union: ["NodeA", "NodeB", "NodeC"]
# acc becomes ["NodeA", "NodeB", "NodeC"].
# Index 2-4: No intersection, acc remains ["NodeA", "NodeB", "NodeC"].
# Result after sort: ["NodeA", "NodeB", "NodeC"].
# This goes on for each row
# Intermediate combined results:
#  combinedResult = [["NodeA", "NodeB", "NodeC"], ["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeD", "NodeE", "NodeF"], ["NodeG"]]
# WITH distinct(combinedResult) as combinedResult
# combinedResult = [["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeG"]]

query_part2 = """
UNWIND $results as result
WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size($results)-1, 1) |
    CASE WHEN result <> $results[index2] AND
        size(apoc.coll.intersection(acc, $results[index2])) > 0
        THEN apoc.coll.union(acc, $results[index2])
        ELSE acc
    END
)) as combinedResult
WITH distinct(combinedResult) as combinedResult
RETURN combinedResult
"""

# Execute the second part of the query with results from the first part
results_part2 = run_query_with_params(driver, query_part2, {'results': flat_results})

# Print the results from the second part
print("Results from the second part:")
for result in results_part2:
    print(result[0])

# Define the third part of the Cypher query to further process combined results
# Given combinedResults as:
# combinedResults = [["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeA", "NodeB"], ["NodeG"], ["NodeD", "NodeE"]]
# Inital list
# allCombinedResults = [["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeA", "NodeB"], ["NodeG"], ["NodeD", "NodeE"]]
# Iteration over each index:
# Index 0: combinedResult = ["NodeA", "NodeB", "NodeC"]
#   No list fully contains ["NodeA", "NodeB", "NodeC"].
#   Include this list in the result.
# Index 1: combinedResult = ["NodeD", "NodeE", "NodeF"]
#   No list fully contains ["NodeD", "NodeE", "NodeF"].
#   Include this list in the result.
# Index 2: combinedResult = ["NodeA", "NodeB"]
#   The list at index 0 ["NodeA", "NodeB", "NodeC"] fully contains ["NodeA", "NodeB"].
#   Exclude this list from the result.
# Index 3: combinedResult = ["NodeG"]
#   No list fully contains ["NodeG"].
#   Include this list in the result.
# Index 4: combinedResult = ["NodeD", "NodeE"]
#   The list at index 1 ["NodeD", "NodeE", "NodeF"] fully contains ["NodeD", "NodeE"].
#   Exclude this list from the result.
# Final Result:
#   combinedResult = [["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeG"]]

query_part3 = """
WITH $combinedResults AS combinedResults
WITH collect(combinedResults) as allCombinedResults
UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
    WHERE x <> combinedResultIndex
    AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
)
RETURN combinedResult
"""

# Execute the third part of the query with results from the second part
results_part3 = run_query_with_params(driver, query_part3, {'combinedResults': [r[0] for r in results_part2]})

# Print the results from the third part
print("Results from the third part:")
print(results_part3[0][0])

# Define the fourth part of the Cypher query to return nodes
# We loop through this for each item in below list
# combinedResults = [["NodeA", "NodeB", "NodeC"], ["NodeD", "NodeE", "NodeF"], ["NodeG"]]
# First Iteration (Subquery):
# combinedResult = ["NodeA", "NodeB", "NodeC"]
# UNWIND: Expands to:
# name = "NodeA"
# name = "NodeB"
# name = "NodeC"
# MATCH (e:Entity {name}):
# e = Node with name "NodeA"
# e = Node with name "NodeB"
# e = Node with name "NodeC"
# ORDER BY size(e.name) DESC: Ensured the node with descriptive name is at the top.
# RETURN collect(e) AS nodes
# nodes = [NodeA, NodeB, NodeC]
# CALL apoc.refactor.mergeNodes(nodes, {properties: { .*: 'discard' }}):
# Merges NodeA, NodeB, and NodeC into one node, discarding all properties.
# The other options are:
#   discard: Discard the properties from the source nodes. The resulting node will not have these properties.
#   overwrite: Overwrite the properties in the target node with those from the source nodes.
#   combine: Combine the properties from all source nodes. If there are conflicts, properties from the later nodes in the list will overwrite those from earlier nodes.

query_part4_return_nodes = """
CALL {
  WITH $combinedResults AS combinedResult
  UNWIND combinedResult AS name
  MATCH (e:__Entity__ {name:name})
  WITH e
  ORDER BY size(e.name) DESC // prefer longer names to remain after merging
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
"""
# [['Epic', 'Epic Games'], ['Bank of America', 'Bank of America Corp.'], ['Star Ocean: The Second Story R', 'Star Ocean: The Second Story R logos'], ['logo leak', 'logo leaked'], ['Star Ocean: First Departure', 'Star Ocean: First Departure R'], ['support.na.square-enix.com/images/title_banner/title_banner_19285.jpg', 'support.na.square-enix.com/images/title_banner/title_banner_19288.jpg'], ['FASTag', 'Fastag'], ['333-228379', '333-228379-04']]
# Execute the fourth part of the query. Loop for each similar nodes
merge_nodes = []
for r in results_part3[0][0]:
    merge_nodes.append(r)

print("merged ", merge_nodes)
pots_merged_nodes = []
for merge_node in merge_nodes:
    results_part4_nodes = run_query_with_params(driver, query_part4_return_nodes, {'combinedResults': merge_node})
    pots_merged_nodes.append(results_part4_nodes)

# Print the results from the fourth part
print("Results from the fourth part (nodes):")
for result in pots_merged_nodes:
    print(result)

print(len(pots_merged_nodes))

# Close the driver
driver.close()

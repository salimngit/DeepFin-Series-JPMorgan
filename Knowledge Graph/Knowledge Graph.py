# Databricks notebook source
import os
import pandas as pd
import spacy 
import requests
import operator
import uuid
import numpy as np
from collections import Counter 

model_dir = spacy.util.get_data_path() 
if not os.path.exists(os.path.join(model_dir.as_posix(), "en")): 
  spacy.cli.download("en")
 
nlp = spacy.load("en")

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

def get_entities_udf():
    def get_entities(text):
        global nlp        
        doc = nlp(str(text))
        
        return [t.text for t in doc.ents]
    res_udf = udf(get_entities, ArrayType(StringType()))
    return res_udf

def get_pos_chain_udf():
  def get_pos_chain(text):
        global nlp        
        doc = nlp(str(text))
                    
        return "-".join([d.tag_ for d in doc]) 
  res_udf = udf(get_pos_chain, StringType())
  return res_udf
  

def get_verb(token, omittdescription):
    """Check verb type given spacy token"""
    
    if token.pos_ == 'VERB':
        indirect_object = False
        direct_object = False
        for item in token.children:
            if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                indirect_object = True
            if (item.dep_ == "dobj" or item.dep_ == "dative"):
                direct_object = True
        if indirect_object and direct_object:
            description = 'DITRANVERB'
            token_text = token.text
        elif direct_object and not indirect_object:
            description = 'TRANVERB'
            token_text = token.text
        elif not direct_object and not indirect_object:
            description = 'INTRANVERB'
            token_text = token.text
        else:
            description = 'VERB'
            token_text = token.text    
        
        #return based on function settings
        if omittdescription :
            return token_text
        else:
            return (description, token_text)
            
def get_verbs_udf():
  def get_verbs(title_text,omittdescription=True):    
    global nlp        
    doc = nlp(str(title_text))
    
    verbs = []
    for token in doc:
        verb = get_verb(token, omittdescription)
        if verb is not None:
            verbs.append(verb)
    return verbs
  
  res_udf = udf(get_verbs, ArrayType(StringType()))
  return res_udf

def get_concept_probability(entity_name):
    response = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={}&topK=5' \
                            .format(entity_name))
    
    if response.status_code == 200:
        if len(response.json().items()) > 0 :
            concept_dict = response.json() #assign response to variable to use as dictionary            
            max_concept = sorted(concept_dict.items(), key=operator.itemgetter(1), reverse=True)[0]                        
            return [max_concept[0], format(max_concept[1], '.2f')]
            
    return None
  
uuidUdf= udf(lambda : str(uuid.uuid4()),StringType()).asNondeterministic()

# COMMAND ----------

df = spark.sql("SELECT *  FROM uci_news_aggregator LIMIT 100")
df.cache()
df.count()

# COMMAND ----------

df = df.withColumn("title_pos_chain", get_pos_chain_udf()("TITLE"))
df = df.withColumn("named_entities", get_entities_udf()("TITLE"))
df = df.withColumn("title_verbs", get_verbs_udf()("TITLE"))

df.cache()
df.show()

# COMMAND ----------

def save_lookup_tables(lookup_fields_dict):
  # save all key lookup entities in tables for easy search during graph build
  for key, value in lookup_fields_dict.items():  
    lookup_schema = StructType([StructField(key, StringType(), True)])
    lookup_rdd = sc.parallelize(value)
    lookup_df = spark.createDataFrame(lookup_rdd, lookup_schema)
    lookup_df.write.saveAsTable(key, mode="overwrite")

def get_unique_item_udf():    
  def matching_functioning(entity_query, table_name):
    fuzzysearch_df = spark.sql("SELECT * from {}".format(table_name))
    return "test"

  '''
    fuzzysearch_df = (fuzzysearch_df \
        .withColumn("named_entities", lower(col(table_name))) \
        .withColumn("query", lit(entity_query)) \
        .withColumn("len_query", length(col("query"))) \
        .withColumn("len_" + table_name, length(col(table_name))) \
        .withColumn("edit_distance", levenshtein(table_name, "query")) \
        .withColumn("max_length", when( (col("len_query") > col("len_" + table_name)), \
                                         col("len_query")).otherwise(col("len_" + table_name))) \
        .withColumn("similarity_ratio", (col("max_length") - col("edit_distance")) / col("max_length") ) \
        .select("*").where(col("similarity_ratio") >= 0.5).orderBy("similarity_ratio", length(table_name), ascending=False)
        )
    # get closest otherwise add to search space
    if fuzzysearch_df.count() > 0:
      #print('found')
      return fuzzysearch_df.first()[0]
    else:
      #print('not found, appending')
      new_entity_df = spark.sql("SELECT '{}' as {}".format(entity_query, table_name))
      new_entity_df.write.mode("append").saveAsTable(table_name)
      return entity_query'''

    
  res_udf = udf(matching_functioning, StringType())
  return res_udf
  

# COMMAND ----------

# extract core entities
publishers = df.select("PUBLISHER").distinct().withColumn("publisher_id", uuidUdf()).select("publisher_id","PUBLISHER")
timestamps = df.select("TIMESTAMP").distinct().withColumn("Date", to_date(from_unixtime(col("TIMESTAMP") / 1000))).select("Date").distinct().withColumn("date_id", uuidUdf()).select("date_id","Date")

named_entities = df.withColumn("entities", explode("named_entities")).select(lower(col("entities"))).distinct().collect()
lookup_fields_dict = {"named_entities":named_entities}
save_lookup_tables(lookup_fields_dict)

#publishers.persist()
#publishers.show()
#timestamps.show()

# COMMAND ----------

pos_chain_1 = "NNP-VBZ-NNP"

get_entity_type_udf = udf(get_concept_probability, ArrayType(StringType()))

base_df = (df
           .where(df["title_pos_chain"].contains(pos_chain_1))
           .filter(size(df["named_entities"]) == 2)
           .withColumn("event_id", uuidUdf() )           
           .withColumn("named_entity1", col("named_entities")[0] )
           .withColumn("named_entity2", col("named_entities")[1] )
           .withColumn("named_entity1_id", uuidUdf() )
           .withColumn("named_entity2_id", uuidUdf() )
           .withColumn("named_entity1_type", get_entity_type_udf("named_entity1")[0])
           .withColumn("named_entity1_type_probability", get_entity_type_udf("named_entity1")[1])
           .withColumn("named_entity2_type", get_entity_type_udf("named_entity2")[0])
           .withColumn("named_entity2_type_probability", get_entity_type_udf("named_entity2")[1])
           .withColumn("Date", to_date(from_unixtime(col("TIMESTAMP") / 1000)))
           .select("ID","TITLE", "PUBLISHER", "URL", "Date", "event_id", "title_verbs", \
                   "named_entity1_id", "named_entity1","named_entity2","named_entity1_type", "named_entity1_type_probability", \
                   "named_entity2_id", "named_entity2_type","named_entity1_type","named_entity2_type", "named_entity2_type_probability")
           .where(col("named_entity1_type").isNotNull() & col("named_entity2_type").isNotNull())
          )

graphbase_df = base_df.join(publishers, ['PUBLISHER'] )
graphbase_df = graphbase_df.join(timestamps, ['Date'] )
graphbase_df.cache()
graphbase_df.printSchema()

# COMMAND ----------

from graphframes import *

#create Nodes
article_nodes = (graphbase_df            
            .withColumn("nodeType", lit("Article"))
            .withColumn("probability", lit(100))
            .select(col("ID").alias("id"),"nodeType",col("URL").alias("value"),"probability")
           )
publisher_nodes = (publishers                   
                   .withColumn("id", col("publisher_id"))
                   .withColumn("nodeType", lit("Publisher"))
                   .withColumn("probability", lit(100))
                   .select("id","nodeType",col("PUBLISHER").alias("value"),"probability")  
                  )

timestamp_nodes = (timestamps                   
                   .withColumn("id", col("date_id"))
                   .withColumn("nodeType", lit("Date"))
                   .withColumn("probability", lit(100))
                   .select("id","nodeType",col("Date").alias("value"),"probability")
                  )

named_entity1_nodes = (graphbase_df
                       .withColumn("id", col("named_entity1_id"))
                       .withColumn("nodeType", col("named_entity1_type"))
                       .select("id","nodeType",col("named_entity1").alias("value"), col("named_entity1_type_probability").alias("probability"))
                      )

named_entity2_nodes = (graphbase_df
                       .withColumn("id", col("named_entity2_id"))
                       .withColumn("nodeType", col("named_entity2_type"))
                       .select("id","nodeType",col("named_entity2").alias("value"), col("named_entity2_type_probability").alias("probability"))
                      )

CVT_event_nodes = (graphbase_df
                       .withColumn("id", col("event_id"))
                       .withColumn("nodeType", lit("@Event"))
                       .withColumn("probability", lit(100))
                       .select("id","nodeType",col("title_verbs").cast("string").alias("value"), "probability")
                       .where(col("title_verbs").isNotNull())
                      )

article_publisher_edges = (graphbase_df
                           .select(col("ID").alias("src"), col("publisher_id").alias("dst"),col("URL").alias("value"))
                          )

article_timestamp_edges = (graphbase_df
                           .select(col("ID").alias("src"), col("date_id").alias("dst"),col("Date").alias("value"))
                          )

event_named_entity1_edges = (graphbase_df
                             .select(col("named_entity1_id").alias("src"), col("event_id").alias("dst"), col("title_verbs").cast("string").alias("value") ) 
                            )

event_named_entity2_edges = (graphbase_df
                             .select(col("named_entity2_id").alias("src"), col("event_id").alias("dst"), col("title_verbs").cast("string").alias("value") ) 
                            )

named_entity_nodes = named_entity1_nodes.union(named_entity2_nodes).union(CVT_event_nodes)
event_named_entity_edges = event_named_entity1_edges.union(event_named_entity2_edges)

all_nodes = article_nodes.union(publisher_nodes).union(timestamp_nodes).union(named_entity_nodes)
all_edges = article_publisher_edges.union(article_timestamp_edges).union(event_named_entity_edges)

all_nodes.write.saveAsTable("graph_nodes", mode="overwrite")
all_edges.write.saveAsTable("graph_edges", mode="overwrite")

all_nodes.cache()
all_edges.cache()

all_graph = GraphFrame(all_nodes, all_edges)

# COMMAND ----------

EDA_df = spark.sql("SELECT nodeType, COUNT(*) FROM graph_nodes GROUP BY nodeType")
display(EDA_df)

# COMMAND ----------

# entity disambiguity 

standard_nodes = "'Publisher', 'Article', 'site', 'Date', '@Event'"

named_entities_df = spark.sql("SELECT * from graph_nodes WHERE nodeType not in (" + standard_nodes + ")")

named_entities_df = named_entities_df.crossJoin(named_entities_df.withColumnRenamed("value" , "value2").withColumnRenamed("id" , "id2").select("value2","id2"))

named_entities_df = (named_entities_df
                     .withColumn("value", lower(col("value")))
                     .withColumn("value2", lower(col("value2")))
                     .withColumn("len_value", length(col("value")))
                     .withColumn("len_value2", length(col("value2")))
                     .withColumn("edit_distance", levenshtein("value", "value2"))
                     .withColumn("max_length", when( (col("len_value") > col("len_value2")), \
                                     col("len_value")).otherwise(col("len_value2")))
                     .withColumn("similarity_ratio", (col("max_length") - col("edit_distance")) / col("max_length") )                     
                     .where((col("similarity_ratio") >= 0.6) & (col("id") != col("id2")) )
                    )

named_entities_df.show()                       

# COMMAND ----------

from urllib.parse import quote

def urlencode(value):
  return quote(value, safe="")


udf_urlencode = udf(urlencode, StringType())

def to_cosmosdb_vertices(dfVertices, labelColumn, partitionKey = ""):
  dfVertices = dfVertices.withColumn("id", udf_urlencode("id"))
  
  columns = ["id", labelColumn]
  
  if partitionKey:
    columns.append(partitionKey)
  
  columns.extend(['nvl2({x}, array(named_struct("id", uuid(), "_value", {x})), NULL) AS {x}'.format(x=x) \
                for x in dfVertices.columns if x not in columns])
 
  return dfVertices.selectExpr(*columns).withColumnRenamed(labelColumn, "label")

def to_cosmosdb_edges(g, labelColumn, partitionKey = ""): 
  dfEdges = g.edges
  
  if partitionKey:
    dfEdges = dfEdges.alias("e") \
      .join(g.vertices.alias("sv"), col("e.src") == col("sv.id")) \
      .join(g.vertices.alias("dv"), col("e.dst") == col("dv.id")) \
      .selectExpr("e.*", "sv." + partitionKey, "dv." + partitionKey + " AS _sinkPartition")

  dfEdges = dfEdges \
    .withColumn("id", udf_urlencode(concat_ws("_", col("src"), col(labelColumn), col("dst")))) \
    .withColumn("_isEdge", lit(True)) \
    .withColumn("_vertexId", udf_urlencode("src")) \
    .withColumn("_sink", udf_urlencode("dst")) \
    .withColumnRenamed(labelColumn, "label") \
    .drop("src", "dst")
  
  return dfEdges


cosmosDbVertices = to_cosmosdb_vertices(all_graph.vertices, "nodeType")
#display(cosmosDbvertices)

cosmosDbEdges = to_cosmosdb_edges(all_graph,"value")
display(cosmosDbEdges)

# COMMAND ----------

#insert into CosmosDB

cosmosDbConfig = {
  "Endpoint" : "https://salimncosmosdb.documents.azure.com:443/",
  "Masterkey" : "dEx9D4AEZ84qOoKRxbdSJNaTOU99fOdmHbP8YjRWmGqQavRTcxiXg6v5F90iB71UuUj26DnLamxn7I1sZtS6UA==",
  "Database" : "DeepFin",
  "Collection" : "KnowledgeGraph",
  "Upsert" : "true"
}

cosmosDbFormat = "com.microsoft.azure.cosmosdb.spark"

cosmosDbVertices.write.format(cosmosDbFormat).mode("append").options(**cosmosDbConfig).save()
cosmosDbEdges.write.format(cosmosDbFormat).mode("append").options(**cosmosDbConfig).save()


# COMMAND ----------



# COMMAND ----------

#display(all_graph.vertices.filter("nodeType = 'Date'"))
display(all_graph.inDegrees.orderBy(desc("inDegree")).limit(5))

# COMMAND ----------

not surfrom pyspark.sql.functions import size

pos_chain_1 = "NNP-VBZ-NNP"

graphbase_df = (df
                .where(df["title_pos_chain"].contains(pos_chain_1))
                .filter(size(df["named_entities"]) == 2)).rdd.collect()

graphbase_list = [{'TITLE' : x["TITLE"], 'named_entities' : x["named_entities"], 'title_verbs' : x["title_verbs"]}
                  for x in graphbase_df]

for r in graphbase_list:    
  print (r["TITLE"])
  print ('named entities:{} verbs:{}'.format(r["named_entities"], r["title_verbs"]))  
  print ('entity types: {}'.format([ get_concept_probability(entity_name) for entity_name in r["named_entities"]] ))
  print ()    


# COMMAND ----------

#spacy.explain('VBP')
#print(get_concept_probability('Carl Icahn'))

# COMMAND ----------


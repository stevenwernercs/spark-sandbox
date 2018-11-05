package earth.trifid.example

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
//package org.apache.spark.examples.ml


//https://github.com/soundcloud/cosine-lsh-join-spark

// $example on$
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
// $example off$
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object TfIdfExample {

  def featurize(rawContent: DataFrame): DataFrame = {

    //TOKENIZE
    val tokenizer = new Tokenizer().setInputCol("raw").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)
    //val wordsData = tokenizer.transform(sentenceData)
    val wordsData = regexTokenizer.transform(rawContent)

    //STOPWORD REMOVER
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filteredWords")
    val filteredWordsData = remover.transform(wordsData)

    //TF
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(filteredWordsData)

    featurizedData
  }


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("TfIdfExample").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .getOrCreate()

    // $example on$
    val dataSet = spark.read.option("sep", "\t").option("header", "true").csv("data.tsv")
    val querySet = spark.read.option("sep", "\t").option("header", "true").csv("query.tsv")

    val featurizedData = featurize(dataSet)
    val featurizedQueries = featurize(querySet)

    //TD model generation
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    //TFIDF
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.select("id", "features").take(3).foreach(println)

    val rescaledQueries = idfModel.transform(featurizedQueries)
    //rescaledQueries.select("id", "features").take(3).foreach(println)

    val mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = mh.fit(rescaledData)

    // Feature Transformation
    //println("The hashed dataset where hashed values are stored in the column 'hashes':")
    //val dataHashes = model.transform(rescaledData)

    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    println("Approximately joining dfA and dfB on Jaccard distance smaller than 0.6:")
    val results = model.approxSimilarityJoin(rescaledData, rescaledQueries, 1, "JaccardDistance")
      .select(col("datasetA.id").alias("id-data"),
        col("datasetB.id").alias("id-query"),
        col("JaccardDistance"))

    //WANT, for each search provide sorted rank data id
    results.orderBy(asc("id-query"), asc("JaccardDistance")).show(false)

    // $example off$

    spark.stop()
  }
}
// scalastyle:on println



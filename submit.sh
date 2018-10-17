#!/bin/bash

spark-submit --class "earth.trifid.WordCounter" --master "local[*]" target/sandbox-1.0-SNAPSHOT-jar-with-dependencies.jar

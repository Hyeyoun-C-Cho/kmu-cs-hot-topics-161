import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StopWordsRemover

object WordCount {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)
    val swr = new StopWordsRemover().getStopWords

    // print input,output path
    print("input file location is: " + args(0) + "\n")
    print("output file location is: " + args(1) + "\n")

    // get textfile
    val tweets = sc.textFile(args(0))

    // split srting, make srting to lowercase
    val tweet_words = tweets.flatMap(x => x.toLowerCase().split("\\s+"))
    // filtering words whose length is equal or more than four characters 
    val tweet_wf = tweet_words.filter(x => x.length>=4)
    // remove special characters and make it empty 
    val remove_sc = tweet_wf.map(_.replaceAll("[~!@#$^%&*\\(\\)_+={}\\[\\]|;:\"<,>.?'/\\\\-]", ""))
    // filtering nonempty words
    val remove_em = remove_sc.filter(_.nonEmpty)

    // remove stopwords
    val stop_words = remove_em.filter(removeword => !swr.contains(removeword))

    // make map/reduce with stop_words
    val kv = stop_words.map(x => (x, 1))
    val word_count = kv.reduceByKey((x,y) => x+y)

    // get top words
    val top = word_count.takeOrdered(10)(Ordering[Int].reverse.on(x=>x._2))
    val rdd = sc.parallelize(top)
    rdd.saveAsTextFile(args(1))
    top.foreach(println)
  }
}

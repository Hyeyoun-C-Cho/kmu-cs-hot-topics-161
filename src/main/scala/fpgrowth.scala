import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD


object fpgrowth {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("FP Growth")
    val sc = new SparkContext(conf)

    val data = sc.textFile(args(0), 24)

    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))

    val fpg = new FPGrowth()
      .setMinSupport(0.3)
      .setNumPartitions(1)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    val itemsetmodel = model.freqItemsets.map { itemset => s"""[${itemset.items.mkString(",")}], ${itemset.freq}""" }

    val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent.mkString("[", ",", "]")
          + ", " + rule.confidence)
    }

    val generatemodel = model.generateAssociationRules(minConfidence).map { rule => s"""[${rule.antecedent.mkString(",")}] => [${rule.consequent.mkString(",")}], ${rule.confidence}""" }

    itemsetmodel.saveAsTextFile(args(1))
    generatemodel.saveAsTextFile(args(2))

  }
}

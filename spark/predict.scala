package StudentCourseDesign


import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object LogisticRegressionModel {
  def main(args: Array[String]): Unit = {

    // 初始化SparkSession
    val spark = SparkSession.builder()
      .appName("Student Logistic Regression")
      .master("local[*]")
      .getOrCreate()

    // 读取本地CSV数据
    val filePath = "hdfs://localhost:9000/student/student.csv"
    val dataDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .csv(filePath)

    // 显示数据结构
    dataDF.printSchema()
    dataDF.show(5)

    // 数据预处理：选择特征列和目标列
    // 假设目标列是 "Target"，特征包括 "Age at enrollment" 和 "Admission grade" 等
    val selectedColumns = Array("Age at enrollment", "Admission grade", "Target")
    val processedDF = dataDF.select(selectedColumns.map(col): _*)
      .na.drop() // 去除缺失值

    // 将目标列编码为数值型
    val labelIndexer = new StringIndexer()
      .setInputCol("Target")
      .setOutputCol("label")
      .fit(processedDF)
    val labeledDF = labelIndexer.transform(processedDF)

    // 将特征列合并为单一向量列
    val featureColumns = Array("Age at enrollment", "Admission grade")
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")
    val featureDF = assembler.transform(labeledDF)

    // 划分训练集和测试集
    val Array(trainingData, testData) = featureDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // 定义逻辑回归模型
    val logisticRegression = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(100)
      .setRegParam(0.01)

    // 训练模型
    val lrModel = logisticRegression.fit(trainingData)

    // 保存模型
    val modelPath = "C:/Users/73269/Desktop/ab/2024/大三下/Web系统设计/大作业/code/GraduationManagementSystem/web/model1"
    lrModel.write.overwrite().save(modelPath)

    println(s"Model saved to: $modelPath")

    // 在测试集上评估模型
    val predictions = lrModel.transform(testData)

    // 显示部分预测结果
    predictions.select("features", "label", "prediction").show(10)

    // 评估模型的准确率
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Test set accuracy = $accuracy")

    // 停止SparkSession
    spark.stop()
  }
}

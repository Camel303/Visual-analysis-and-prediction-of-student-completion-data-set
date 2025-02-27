package StudentCourseDesign

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object GraduationPredictionRandomForest {
  def main(args: Array[String]): Unit = {
    // 初始化 SparkSession
    val spark = SparkSession.builder()
      .appName("Graduation Management System - Random Forest")
      .master("local[*]") // 本地模式
      .getOrCreate()

    // 读取数据
    val filePath = "hdfs://localhost:9000/student/student.csv"
    val dataDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";") // 设置分隔符为分号
      .csv(filePath)

    // 数据预处理：将目标列转换为数值型
    // 假设 "Target" 列表示分类标签（Graduate=1, Dropout=0）
    val processedDF = dataDF.withColumn("label", when(col("Target") === "Graduate", 1.0).otherwise(0.0))

    // 特征向量化：选择用于训练的特征列
    val featureCols = Array(
      "Admission grade",
      "Age at enrollment",
      "Curricular units 1st sem (grade)",
      "Curricular units 2nd sem (grade)"
    ) // 根据数据列选择特征
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val finalDF = assembler.transform(processedDF).select("features", "label")

    // 划分训练集和测试集
    val Array(trainingData, testData) = finalDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // 创建随机森林分类模型
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(20) // 设置随机森林的树的数量，可根据需要调整
      .setMaxDepth(5) // 设置最大深度，可根据需要调整

    // 训练模型
    val rfModel = rf.fit(trainingData)

    // 保存模型到本地
    val modelPath = "C:/Users/73269/Desktop/ab/2024/大三下/Web系统设计/大作业/code/GraduationManagementSystem/web/mode/model3" // 替换为你想要保存模型的实际路径
    rfModel.write.overwrite().save(modelPath)
    println(s"Random Forest Model saved to: $modelPath")

    // 在测试集上进行预测
    val predictions = rfModel.transform(testData)

    // 显示预测结果
    predictions.select("features", "label", "prediction", "probability").show()

    // 模型评估：计算准确率、精准率、召回率和 F1 值
    // 准确率
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = accuracyEvaluator.evaluate(predictions)

    // 精准率 (Precision)
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val precision = precisionEvaluator.evaluate(predictions)

    // 召回率 (Recall)
    val recallEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val recall = recallEvaluator.evaluate(predictions)

    // F1 值
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val f1 = f1Evaluator.evaluate(predictions)

    // 输出评估指标
    println(s"Test set accuracy = $accuracy")
    println(s"Test set precision = $precision")
    println(s"Test set recall = $recall")
    println(s"Test set F1 score = $f1")

    // 停止 SparkSession
    spark.stop()
  }
}

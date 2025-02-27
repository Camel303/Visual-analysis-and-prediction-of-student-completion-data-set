package StudentCourseDesign

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, QuantileDiscretizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object GraduationPredictionNaiveBayes {
  def main(args: Array[String]): Unit = {
    // 初始化 SparkSession
    val spark = SparkSession.builder()
      .appName("Graduation Management System - Naive Bayes")
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

    // 将特征列的类型转换为 DoubleType
    val doubleDF = processedDF
      .withColumn("Admission grade", col("Admission grade").cast("double"))
      .withColumn("Age at enrollment", col("Age at enrollment").cast("double"))
      .withColumn("Curricular units 1st sem (grade)", col("Curricular units 1st sem (grade)").cast("double"))
      .withColumn("Curricular units 2nd sem (grade)", col("Curricular units 2nd sem (grade)").cast("double"))

    // 特征离散化：将连续特征转换为离散特征
    val discretizers = Array(
      new QuantileDiscretizer()
        .setInputCol("Admission grade")
        .setOutputCol("Admission grade_bucket")
        .setNumBuckets(5),
      new QuantileDiscretizer()
        .setInputCol("Age at enrollment")
        .setOutputCol("Age at enrollment_bucket")
        .setNumBuckets(5),
      new QuantileDiscretizer()
        .setInputCol("Curricular units 1st sem (grade)")
        .setOutputCol("Curricular units 1st sem (grade)_bucket")
        .setNumBuckets(5),
      new QuantileDiscretizer()
        .setInputCol("Curricular units 2nd sem (grade)")
        .setOutputCol("Curricular units 2nd sem (grade)_bucket")
        .setNumBuckets(5)
    )

    val discretizedDF = discretizers.foldLeft(doubleDF) { (df, discretizer) =>
      discretizer.fit(df).transform(df)
    }

    // 特征向量化：选择离散化后的特征列
    val featureCols = Array(
      "Admission grade_bucket",
      "Age at enrollment_bucket",
      "Curricular units 1st sem (grade)_bucket",
      "Curricular units 2nd sem (grade)_bucket"
    )

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val finalDF = assembler.transform(discretizedDF).select("features", "label")

    // 划分训练集和测试集
    val Array(trainingData, testData) = finalDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // 创建朴素贝叶斯分类模型
    val nb = new NaiveBayes()
      .setModelType("multinomial") // 设置模型类型为多项式模型（默认值）

    // 训练模型
    val nbModel = nb.fit(trainingData)

    // 保存模型到本地
    val modelPath = "C:/Users/73269/Desktop/ab/2024/大三下/Web系统设计/大作业/code/GraduationManagementSystem/web/mode/model4" // 替换为你想要保存模型的实际路径
    nbModel.write.overwrite().save(modelPath)
    println(s"Naive Bayes Model saved to: $modelPath")

    // 在测试集上进行预测
    val predictions = nbModel.transform(testData)

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

package StudentCourseDesign

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.jpmml.sparkml.PMMLBuilder // 引入 PMMLBuilder
import org.jpmml.model.JAXBUtil
import javax.xml.transform.stream.StreamResult // 引入 StreamResult
import java.io.File // 用于 File 对象

object GraduationPrediction2 {
  def main(args: Array[String]): Unit = {
    // 初始化 SparkSession
    val spark = SparkSession.builder()
      .appName("Graduation Management System with PMML Export")
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
    val processedDF = dataDF.withColumn("label", when(col("Target") === "Graduate", 1.0).otherwise(0.0))

    // 特征向量化：选择用于训练的特征列
    val featureCols = Array(
      "Admission grade",
      "Age at enrollment",
      "Curricular units 1st sem (grade)",
      "Curricular units 2nd sem (grade)"
    )
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // 创建逻辑回归模型
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // 创建 Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr)) // 将特征处理和模型放入 Pipeline 中

    // 划分训练集和测试集
    val Array(trainingData, testData) = processedDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // 训练模型
    val pipelineModel = pipeline.fit(trainingData) // 返回的是 PipelineModel

    // 使用 PMMLBuilder 将模型转换为 PMML
    val pmml = new PMMLBuilder(trainingData.schema, pipelineModel).build()

    // 保存 PMML 文件到本地
    val pmmlFilePath = "C:/Users/73269/Desktop/ab/2024/大三下/Web系统设计/大作业/code/GraduationManagementSystem/web/mode/GraduationPredictionModel.pmml"
    val pmmlFile = new File(pmmlFilePath) // 创建 File 对象
    val streamResult = new StreamResult(pmmlFile) // 使用 StreamResult 包装 File 对象
    JAXBUtil.marshalPMML(pmml, streamResult) // 将 PMML 写入文件

    println(s"PMML model saved to: $pmmlFilePath")

    // 停止 SparkSession
    spark.stop()
  }
}

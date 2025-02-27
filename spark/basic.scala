package StudentCourseDesign


import org.apache.spark.sql.{SparkSession, DataFrame}
import java.io.PrintWriter

object GraduationManagement {
  def main(args: Array[String]): Unit = {

    // 初始化SparkSession
    val spark = SparkSession.builder()
      .appName("Graduation Management System")
      .master("local[*]") // 本地模式，使用所有可用CPU
      .getOrCreate()

    // 读取数据
    val filePath = "hdfs://localhost:9000/student/student.csv"
    val dataDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";") // 设置分隔符为分号
      .csv(filePath)

    // 显示数据结构
    dataDF.printSchema()
    dataDF.show(5) // 显示前5行数据以确认正确性

    // 保存路径
    val savePath = "C:/Users/73269/Desktop/ab/2024/大三下/Web系统设计/大作业/code/GraduationManagementSystem/web/infoStuCD/json/"

    // 保存数据为JSON文件的函数
    def saveResultsAsJson(df: DataFrame, outputPath: String): Unit = {
      val jsonResult = df.toJSON.collect() // 将DataFrame转换为JSON格式的字符串集合
      val writer = new PrintWriter(outputPath)
      writer.write("[")
      writer.write(jsonResult.mkString(","))
      writer.write("]")
      writer.close()
      println(s"统计结果已保存到: $outputPath")
    }

    // 1. 入学资格成绩分布
    val admissionGradeDistribution = dataDF.groupBy("Previous qualification (grade)").count().orderBy("Previous qualification (grade)")
    admissionGradeDistribution.show()
    saveResultsAsJson(admissionGradeDistribution, savePath + "j1.json")

    // 2. 按性别划分的毕业率
    val genderGraduationRate = dataDF.groupBy("Gender", "Target").count().orderBy("Gender", "Target")
    genderGraduationRate.show()
    saveResultsAsJson(genderGraduationRate, savePath + "j2.json")

    // 3. 特征相关性热力图
    val numericColumns = dataDF.columns.filter(colName => dataDF.schema(colName).dataType.simpleString.matches("int|double|float"))
    val correlationMatrix = numericColumns
      .combinations(2)
      .map(pair => (pair(0), pair(1), dataDF.stat.corr(pair(0), pair(1))))
      .toSeq
    val correlationDF = spark.createDataFrame(correlationMatrix).toDF("Feature1", "Feature2", "Correlation")
    correlationDF.show()
    saveResultsAsJson(correlationDF, savePath + "j3.json")

    // 4. 第一学期课程成绩分布
    val firstSemGradeDistribution = dataDF.groupBy("Curricular units 1st sem (grade)").count().orderBy("Curricular units 1st sem (grade)")
    firstSemGradeDistribution.show()
    saveResultsAsJson(firstSemGradeDistribution, savePath + "j4.json")

    // 5. 按婚姻状况划分的毕业率
    val maritalStatusGraduationRate = dataDF.groupBy("Marital status", "Target").count().orderBy("Marital status", "Target")
    maritalStatusGraduationRate.show()
    saveResultsAsJson(maritalStatusGraduationRate, savePath + "j5.json")

    // 6. 入学成绩与第一学期课程成绩的关系
    val admissionVsFirstSemGrade = dataDF.select("Admission grade", "Curricular units 1st sem (grade)", "Target")
    admissionVsFirstSemGrade.show()
    saveResultsAsJson(admissionVsFirstSemGrade, savePath + "j6.json")

    // 停止SparkSession
    spark.stop()
  }
}

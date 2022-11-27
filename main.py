
import pandas
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row


spark = SparkSession.builder.getOrCreate()

Diamonds = pandas.read_csv('/Users/carstenjuliansavage/Desktop/R Working Directory/diamonds.csv')

Diamonds_Spark = spark.createDataFrame(Diamonds)

Diamonds_Spark.show()

Diamonds_Spark.printSchema()

Diamonds_Spark.show(5)

Diamonds_Spark.show(2, vertical=True)

(Diamonds_Spark
 .select(["x", "y", "z"])
 .describe()
 .show()
 )

Diamonds_Spark.collect()

#Diamonds_Pandas = Diamonds_Spark.toPandas()

Diamonds_Spark.select(Diamonds_Spark.carat).show()
import numpy
Diamonds_Spark.withColumn('carat2', 2*(Diamonds_Spark.carat)).show()

Diamonds_Spark.filter(Diamonds_Spark.x > 4).show()

Diamonds_Spark.groupby(['carat','cut']).avg().show()

Diamonds_Spark.createOrReplaceTempView("DIASPARK")
spark.sql("SELECT count(*) from DIASPARK").show()
spark.sql("SELECT * from DIASPARK WHERE CUT LIKE 'I%'").show()



Orders = pandas.read_csv('/Users/carstenjuliansavage/Desktop/R Working Directory/r-sql-demo-files/orders.csv')
Employees = pandas.read_csv('/Users/carstenjuliansavage/Desktop/R Working Directory/r-sql-demo-files/employees.csv')

Orders_Spark = spark.createDataFrame(Orders)
Employees_Spark = spark.createDataFrame(Employees)
Orders_Spark.show()
Employees_Spark.show()
Orders_Spark.createOrReplaceTempView("Orders_Spark")
Employees_Spark.createOrReplaceTempView("Employees_Spark")

EmployeeOrders_Spark = spark.sql("SELECT * from Orders_Spark a LEFT JOIN Employees_Spark b ON a.id = b.id")

EmployeeOrders_Spark.show()

EmployeeOrders = EmployeeOrders_Spark.toPandas()


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()

print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()

print("Spearman correlation matrix:\n" + str(r2[0]))


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler


Diamonds2 = (Diamonds
    .filter(['x','y','z'])
    )

Diamonds2_SP = spark.createDataFrame(Diamonds2)

assembler = VectorAssembler(
    inputCols=["x", "y", "z"],
    outputCol="features")

output = assembler.transform(Diamonds2_SP)
output.show()

r1 = Correlation.corr(output,"features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(output, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))


Diamonds3 = Diamonds.copy()
import numpy
Diamonds3['Dummy'] = numpy.where(Diamonds3['clarity']=="VVS2",'1','0')
Diamonds3.dtypes
Diamonds3['Dummy'] = Diamonds3['Dummy'].astype('float64')

Diamonds3 = (Diamonds3
 .filter(['x','y','z','Dummy'])
)

Diamonds3_SP = spark.createDataFrame(Diamonds3)

assembler = VectorAssembler(
    inputCols=["x", "y", "z"],
    outputCol="features")

assembler_label = VectorAssembler(
    inputCols=["Dummy"],
    outputCol="label")

output = assembler.transform(Diamonds3_SP)
#output = assembler_label.transform(output)
output.show()
output.dtypes

output = output.select('features','Dummy')
output = output.withColumnRenamed("Dummy","label")

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
r = ChiSquareTest.test(output, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))




from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])
df.show()
df.dtypes

r = ChiSquareTest.test(df, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))





from pyspark.ml.stat import Summarizer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext

# create summarizer for multiple metrics "mean" and "count"
summarizer = Summarizer.metrics("count", "max", "mean", "min", "normL1", "normL2", "numNonZeros", "std", "sum", "variance")
# compute statistics for multiple metrics without weight
df.select(summarizer.summary(df.features)).show(truncate=False)
# compute statistics for single metric "mean" without weight
df.select(Summarizer.mean(df.features)).show(truncate=False)


# create summarizer for multiple metrics "mean" and "count"
summarizer = Summarizer.metrics("count", "max", "mean", "min", "normL1", "normL2", "numNonZeros", "std", "sum", "variance")
# compute statistics for multiple metrics without weight
output.select(summarizer.summary(output.features)).show(truncate=False)
# compute statistics for single metric "mean" without weight
output.select(Summarizer.mean(output.features)).show(truncate=False)

Output_Pandas = output.toPandas()
Output_Pandas.describe()
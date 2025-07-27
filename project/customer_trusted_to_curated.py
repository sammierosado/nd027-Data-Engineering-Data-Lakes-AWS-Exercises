import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# @params: [JOB_NAME, GLUE_DATABASE_NAME, GLUE_CUSTOMER_TRUSTED_TABLE, GLUE_ACCELEROMETER_TRUSTED_TABLE, S3_OUTPUT_PATH]
# GLUE_DATABASE_NAME example: stedi_data_lake
# GLUE_CUSTOMER_TRUSTED_TABLE example: customer_trusted
# GLUE_ACCELEROMETER_TRUSTED_TABLE example: accelerometer_trusted
# S3_OUTPUT_PATH example: s3://stedi-data-lake-sammie-project/customers_curated/
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'GLUE_DATABASE_NAME',
    'GLUE_CUSTOMER_TRUSTED_TABLE',
    'GLUE_ACCELEROMETER_TRUSTED_TABLE',
    'S3_OUTPUT_PATH'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read customer trusted data using Glue Data Catalog table
customer_trusted_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_CUSTOMER_TRUSTED_TABLE'],
    transformation_ctx="customer_trusted_source"
)

# Read accelerometer trusted data using Glue Data Catalog table
accelerometer_trusted_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_ACCELEROMETER_TRUSTED_TABLE'],
    transformation_ctx="accelerometer_trusted_source"
)

# Convert DynamicFrames to Spark DataFrames for SQL operations
customer_trusted_df = customer_trusted_dyf.toDF()
accelerometer_trusted_df = accelerometer_trusted_dyf.toDF()

# Create temporary views for SQL queries
customer_trusted_df.createOrReplaceTempView("customer_trusted")
accelerometer_trusted_df.createOrReplaceTempView("accelerometer_trusted")

# Join customer_trusted with accelerometer_trusted to find customers who have accelerometer data
# and are already in the trusted list (meaning they agreed to share for research).
# Select distinct customer records.
customers_curated_df = spark.sql("""
    SELECT DISTINCT
        cust.serialNumber,
        cust.shareWithPublicAsOfDate,
        cust.birthday,
        cust.registrationDate,
        cust.shareWithResearchAsOfDate,
        cust.customerName,
        cust.email,
        cust.lastUpdateDate,
        cust.phone,
        cust.shareWithFriendsAsOfDate
    FROM
        customer_trusted cust
    INNER JOIN
        accelerometer_trusted acc ON cust.email = acc.user
""")

# Convert Spark DataFrame back to DynamicFrame
customers_curated_dyf = DynamicFrame.fromDF(customers_curated_df, glueContext, "customers_curated_dyf")

# Write the curated customer data to the curated S3 zone in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=customers_curated_dyf,
    connection_type="s3",
    format="parquet",
    connection_options={"path": args['S3_OUTPUT_PATH'], "partitionKeys": []},
    transformation_ctx="customers_curated_sink"
)

job.commit()

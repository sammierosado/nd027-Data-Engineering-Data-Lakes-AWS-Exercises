import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# @params: [JOB_NAME, GLUE_DATABASE_NAME, GLUE_ACCELEROMETER_LANDING_TABLE, GLUE_CUSTOMER_TRUSTED_TABLE, S3_OUTPUT_PATH]
# GLUE_DATABASE_NAME example: stedi_data_lake
# GLUE_ACCELEROMETER_LANDING_TABLE example: accelerometer_landing
# GLUE_CUSTOMER_TRUSTED_TABLE example: customer_trusted
# S3_OUTPUT_PATH example: s3://stedi-data-lake-sammie-project/accelerometer_trusted/
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'GLUE_DATABASE_NAME',
    'GLUE_ACCELEROMETER_LANDING_TABLE',
    'GLUE_CUSTOMER_TRUSTED_TABLE',
    'S3_OUTPUT_PATH'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read accelerometer data from the landing zone using Glue Data Catalog table
accelerometer_landing_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_ACCELEROMETER_LANDING_TABLE'],
    transformation_ctx="accelerometer_landing_source"
)

# Read customer trusted data using Glue Data Catalog table
customer_trusted_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_CUSTOMER_TRUSTED_TABLE'],
    transformation_ctx="customer_trusted_source"
)

# Convert DynamicFrames to Spark DataFrames for SQL operations
accelerometer_landing_df = accelerometer_landing_dyf.toDF()
customer_trusted_df = customer_trusted_dyf.toDF()

# Create temporary views for SQL queries
accelerometer_landing_df.createOrReplaceTempView("accelerometer_landing")
customer_trusted_df.createOrReplaceTempView("customer_trusted")

# Join accelerometer data with customer_trusted data
# The 'user' field in accelerometer corresponds to the 'email' in customer data
# Explicitly select all columns needed for the next steps, including timeStamp
accelerometer_trusted_df = spark.sql("""
    SELECT
        acc.timeStamp,
        acc.user,
        acc.x,
        acc.y,
        acc.z
    FROM
        accelerometer_landing acc
    INNER JOIN
        customer_trusted cust ON acc.user = cust.email
""")

# Convert Spark DataFrame back to DynamicFrame
accelerometer_trusted_dyf = DynamicFrame.fromDF(accelerometer_trusted_df, glueContext, "accelerometer_trusted_dyf")

# Write the filtered and joined data to the trusted S3 zone in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=accelerometer_trusted_dyf,
    connection_type="s3",
    format="parquet",
    connection_options={"path": args['S3_OUTPUT_PATH'], "partitionKeys": []},
    transformation_ctx="accelerometer_trusted_sink"
)

job.commit()

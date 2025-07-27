import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# @params: [JOB_NAME, GLUE_DATABASE_NAME, GLUE_TABLE_NAME, S3_OUTPUT_PATH]
# GLUE_DATABASE_NAME example: stedi_data_lake
# GLUE_TABLE_NAME example: customer_landing
# S3_OUTPUT_PATH example: s3://stedi-data-lake-sammie-project/customer_trusted/
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'GLUE_DATABASE_NAME', 'GLUE_TABLE_NAME', 'S3_OUTPUT_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read customer data from the landing zone using Glue Data Catalog table
customer_landing_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_TABLE_NAME'],
    transformation_ctx="customer_landing_source"
)

# Convert DynamicFrame to Spark DataFrame
customer_landing_df = customer_landing_dyf.toDF()

# Filter customer records: only keep those who agreed to share data for research
customer_research_df = customer_landing_df.filter(col("shareWithResearchAsOfDate").isNotNull())

# Deduplicate customers by email, keeping the record with the earliest registrationDate
# Define a window specification partitioned by 'email' and ordered by 'registrationDate' ascending
window_spec = Window.partitionBy("email").orderBy(col("registrationDate").asc())

# Add a row number to each partition
customer_deduplicated_df = customer_research_df.withColumn("row_num", row_number().over(window_spec))

# Filter to keep only the first row for each email (which corresponds to the earliest registrationDate)
customer_trusted_df = customer_deduplicated_df.filter(col("row_num") == 1).drop("row_num")

# Convert Spark DataFrame back to DynamicFrame
customer_trusted_dyf = DynamicFrame.fromDF(customer_trusted_df, glueContext, "customer_trusted_dyf")

# Write the filtered and deduplicated data to the trusted S3 zone in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=customer_trusted_dyf,
    connection_type="s3",
    format="parquet",
    connection_options={"path": args['S3_OUTPUT_PATH'], "partitionKeys": []},
    transformation_ctx="customer_trusted_sink"
)

job.commit()

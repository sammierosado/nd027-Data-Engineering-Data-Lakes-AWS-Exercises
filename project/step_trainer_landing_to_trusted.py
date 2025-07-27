import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# @params: [JOB_NAME, GLUE_DATABASE_NAME, GLUE_STEP_TRAINER_LANDING_TABLE, GLUE_CUSTOMERS_CURATED_TABLE, S3_OUTPUT_PATH]
# GLUE_DATABASE_NAME example: stedi_data_lake
# GLUE_STEP_TRAINER_LANDING_TABLE example: step_trainer_landing
# GLUE_CUSTOMERS_CURATED_TABLE example: customers_curated
# S3_OUTPUT_PATH example: s3://stedi-data-lake-sammie-project/step_trainer_trusted/
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'GLUE_DATABASE_NAME',
    'GLUE_STEP_TRAINER_LANDING_TABLE',
    'GLUE_CUSTOMERS_CURATED_TABLE',
    'S3_OUTPUT_PATH'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read step trainer data from the landing zone using Glue Data Catalog table
step_trainer_landing_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_STEP_TRAINER_LANDING_TABLE'],
    transformation_ctx="step_trainer_landing_source"
)

# Read customers curated data using Glue Data Catalog table
customers_curated_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_CUSTOMERS_CURATED_TABLE'],
    transformation_ctx="customers_curated_source"
)

# Convert DynamicFrames to Spark DataFrames for SQL operations
step_trainer_landing_df = step_trainer_landing_dyf.toDF()
customers_curated_df = customers_curated_dyf.toDF()

# Create temporary views for SQL queries
step_trainer_landing_df.createOrReplaceTempView("step_trainer_landing")
customers_curated_df.createOrReplaceTempView("customers_curated")

# Join step_trainer_landing with customers_curated on serialNumber
# This ensures that only step trainer data for curated customers is included.
step_trainer_trusted_df = spark.sql("""
    SELECT
        st.sensorReadingTime,
        st.serialNumber,
        st.distanceFromObject
    FROM
        step_trainer_landing st
    INNER JOIN
        customers_curated cc ON st.serialNumber = cc.serialNumber
""")

# Convert Spark DataFrame back to DynamicFrame
step_trainer_trusted_dyf = DynamicFrame.fromDF(step_trainer_trusted_df, glueContext, "step_trainer_trusted_dyf")

# Write the filtered and joined data to the trusted S3 zone in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=step_trainer_trusted_dyf,
    connection_type="s3",
    format="parquet",
    connection_options={"path": args['S3_OUTPUT_PATH'], "partitionKeys": []},
    transformation_ctx="step_trainer_trusted_sink"
)

job.commit()

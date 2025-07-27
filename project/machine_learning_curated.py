import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType

# @params: [JOB_NAME, GLUE_DATABASE_NAME, GLUE_STEP_TRAINER_TRUSTED_TABLE, GLUE_ACCELEROMETER_TRUSTED_TABLE, GLUE_CUSTOMERS_CURATED_TABLE, S3_OUTPUT_PATH]
# GLUE_DATABASE_NAME example: stedi_data_lake
# GLUE_STEP_TRAINER_TRUSTED_TABLE example: step_trainer_trusted
# GLUE_ACCELEROMETER_TRUSTED_TABLE example: accelerometer_trusted
# GLUE_CUSTOMERS_CURATED_TABLE example: customers_curated
# S3_OUTPUT_PATH example: s3://stedi-data-lake-sammie-project/machine_learning_curated/
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'GLUE_DATABASE_NAME',
    'GLUE_STEP_TRAINER_TRUSTED_TABLE',
    'GLUE_ACCELEROMETER_TRUSTED_TABLE',
    'GLUE_CUSTOMERS_CURATED_TABLE', # Added this parameter for the join logic
    'S3_OUTPUT_PATH'
])
z
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define explicit schema for accelerometer_trusted table
# This is to force the recognition of the 'timestamp' column and ensure robustness
accelerometer_trusted_schema = StructType([
    StructField("timestamp", LongType(), True),
    StructField("user", StringType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("z", DoubleType(), True)
])

# Read step trainer trusted data using Glue Data Catalog table
step_trainer_trusted_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_STEP_TRAINER_TRUSTED_TABLE'],
    transformation_ctx="step_trainer_trusted_source"
)

# Read accelerometer trusted data using Glue Data Catalog table
# Use the explicit schema to read the data
accelerometer_trusted_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_ACCELEROMETER_TRUSTED_TABLE'],
    transformation_ctx="accelerometer_trusted_source",
    push_down_predicate="", # Required when passing schema directly
    additional_options={
        "schema": accelerometer_trusted_schema.json() # Pass schema as JSON string
    }
)

# Read customers curated data using Glue Data Catalog table (NEW INPUT for join logic)
customers_curated_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['GLUE_DATABASE_NAME'],
    table_name=args['GLUE_CUSTOMERS_CURATED_TABLE'],
    transformation_ctx="customers_curated_source"
)

# Convert DynamicFrames to Spark DataFrames for easier manipulation and joining
# Explicitly select fields for step_trainer_trusted_df to ensure schema consistency
step_trainer_trusted_df = step_trainer_trusted_dyf.select_fields(['sensorReadingTime', 'serialNumber', 'distanceFromObject']).toDF()
# For accelerometer_trusted_df, let toDF() infer from the DynamicFrame which used the explicit schema
accelerometer_trusted_df = accelerometer_trusted_dyf.toDF()
customers_curated_df = customers_curated_dyf.toDF() # Convert new input to DataFrame


# --- DEBUGGING: Print schemas to logs ---
print("Schema of step_trainer_trusted_df (Spark DataFrame):")
step_trainer_trusted_df.printSchema()
print("Schema of accelerometer_trusted_df (Spark DataFrame):")
accelerometer_trusted_df.printSchema()
print("Schema of customers_curated_df (Spark DataFrame):")
customers_curated_df.printSchema()
# --- END DEBUGGING ---

# Join step_trainer_trusted with customers_curated to get the email associated with the serialNumber
# This bridges the device ID to the customer email, which is needed to join with accelerometer data
step_trainer_with_email_df = step_trainer_trusted_df.join(
    customers_curated_df,
    step_trainer_trusted_df["serialNumber"] == customers_curated_df["serialNumber"],
    "inner"
).select(
    step_trainer_trusted_df["sensorReadingTime"],
    step_trainer_trusted_df["serialNumber"].alias("step_trainer_serialNumber"),
    step_trainer_trusted_df["distanceFromObject"],
    customers_curated_df["email"].alias("customer_email") # Get the email from curated customers
)

# Now join the result with accelerometer_trusted on timestamp AND email/user
machine_learning_curated_df = step_trainer_with_email_df.join(
    accelerometer_trusted_df,
    (step_trainer_with_email_df["sensorReadingTime"] == accelerometer_trusted_df["timestamp"]) &
    (step_trainer_with_email_df["customer_email"] == accelerometer_trusted_df["user"]), # Join on customer email
    "inner"
).select(
    step_trainer_with_email_df["sensorReadingTime"],
    step_trainer_with_email_df["step_trainer_serialNumber"],
    step_trainer_with_email_df["distanceFromObject"],
    accelerometer_trusted_df["timestamp"].alias("accelerometer_timeStamp"),
    accelerometer_trusted_df["user"].alias("accelerometer_user"),
    accelerometer_trusted_df["x"],
    accelerometer_trusted_df["y"],
    accelerometer_trusted_df["z"]
)

# Convert Spark DataFrame back to DynamicFrame
machine_learning_curated_dyf = DynamicFrame.fromDF(machine_learning_curated_df, glueContext, "machine_learning_curated_dyf")

# Write the curated machine learning data to the curated S3 zone in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=machine_learning_curated_dyf,
    connection_type="s3",
    format="parquet",
    connection_options={"path": args['S3_OUTPUT_PATH'], "partitionKeys": []},
    transformation_ctx="machine_learning_curated_sink"
)

job.commit()
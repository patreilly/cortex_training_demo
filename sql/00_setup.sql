ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:00_setup';
-- Database and schema for the Cortex Training demo.
-- To use a different name, find-and-replace CT_DEMO_DB across sql/, agent/, traces/ and sft/.
CREATE DATABASE IF NOT EXISTS CT_DEMO_DB;
CREATE SCHEMA IF NOT EXISTS CT_DEMO_DB.CORTEX_TRAINING;
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

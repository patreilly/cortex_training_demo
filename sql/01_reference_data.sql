ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:01_reference_data';
-- Synthetic reference data for the support-agent teacher.
-- Fictional SaaS company "Northwind Cloud" selling 6 products.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

CREATE OR REPLACE TABLE PRODUCTS (
    PRODUCT_ID   STRING,
    PRODUCT_NAME STRING,
    DESCRIPTION  STRING
) AS
SELECT * FROM VALUES
    ('P01', 'Northwind Data Sync',   'Scheduled ETL connector that syncs SaaS sources into a warehouse'),
    ('P02', 'Northwind Dashboards',  'Hosted BI dashboards with sharing and scheduled exports'),
    ('P03', 'Northwind Identity',    'SSO, SCIM provisioning, and MFA for Northwind apps'),
    ('P04', 'Northwind API Gateway', 'Rate-limited REST API access with API keys and webhooks'),
    ('P05', 'Northwind Billing',     'Subscriptions, invoices, usage metering, and payment methods'),
    ('P06', 'Northwind Mobile',      'iOS and Android app for viewing dashboards and alerts');

-- SLA rules by tier: first-response target (hours) and whether escalation to engineering is allowed.
CREATE OR REPLACE TABLE SLA_POLICIES (
    TIER                  STRING,
    FIRST_RESPONSE_HOURS  NUMBER,
    RESOLUTION_HOURS      NUMBER,
    CAN_ESCALATE_TO_ENG   BOOLEAN
) AS
SELECT * FROM VALUES
    ('ENTERPRISE', 1, 24, TRUE),
    ('BUSINESS',   4, 72, TRUE),
    ('STARTER',   24, 168, FALSE);

-- 500 customers with deterministic attributes.
CREATE OR REPLACE TABLE CUSTOMERS AS
SELECT
    'C' || LPAD(SEQ4() + 1, 4, '0')                                         AS CUSTOMER_ID,
    ARRAY_CONSTRUCT('Acme','Globex','Initech','Umbrella','Stark','Wayne','Hooli','Vandelay',
                    'Soylent','Cyberdyne','Tyrell','Wonka','Pied Piper','Massive Dynamic',
                    'Oscorp','Gringotts','Monarch','Aperture','Duff','Prestige')[UNIFORM(0,19,RANDOM(11))]::STRING
        || ' ' ||
    ARRAY_CONSTRUCT('Labs','Corp','Retail','Health','Logistics','Bank','Media','Foods','Energy','Systems')[UNIFORM(0,9,RANDOM(12))]::STRING
        || ' ' || (SEQ4() + 1)                                              AS COMPANY_NAME,
    CASE WHEN UNIFORM(1,100,RANDOM(13)) <= 15 THEN 'ENTERPRISE'
         WHEN UNIFORM(1,100,RANDOM(14)) <= 45 THEN 'BUSINESS'
         ELSE 'STARTER' END                                                 AS TIER,
    ARRAY_CONSTRUCT('US','UK','DE','FR','IN','JP','BR','CA','AU')[UNIFORM(0,8,RANDOM(15))]::STRING AS COUNTRY,
    DATEADD(day, -UNIFORM(30, 1500, RANDOM(16)), '2026-11-01'::DATE)        AS CUSTOMER_SINCE,
    ROUND(UNIFORM(1000, 400000, RANDOM(17)), -2)                            AS ANNUAL_CONTRACT_VALUE,
    'csm' || UNIFORM(1, 12, RANDOM(18)) || '@northwind.example'             AS CSM_EMAIL
FROM TABLE(GENERATOR(ROWCOUNT => 500));

-- Escalations created by the CREATE_ESCALATION custom tool.
CREATE TABLE IF NOT EXISTS ESCALATIONS (
    ESCALATION_ID STRING DEFAULT UUID_STRING(),
    TICKET_ID     STRING,
    TEAM          STRING,
    REASON        STRING,
    CREATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

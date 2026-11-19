ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:02_tickets';
-- 3,000 synthetic support tickets. Structured skeleton is deterministic SQL;
-- subject/body text is generated with AI_COMPLETE structured output.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

CREATE OR REPLACE TABLE TICKET_CATEGORIES (PRODUCT_ID STRING, CATEGORY STRING) AS
SELECT * FROM VALUES
    ('P01','sync_failure'),('P01','schema_drift'),('P01','connector_auth'),('P01','slow_sync'),
    ('P02','dashboard_error'),('P02','export_failure'),('P02','sharing_permissions'),('P02','slow_dashboard'),
    ('P03','sso_login_failure'),('P03','scim_provisioning'),('P03','mfa_lockout'),
    ('P04','rate_limit'),('P04','api_key_issue'),('P04','webhook_failure'),
    ('P05','invoice_dispute'),('P05','payment_failed'),('P05','plan_change'),('P05','refund_request'),
    ('P06','app_crash'),('P06','push_notifications');

CREATE OR REPLACE TABLE TICKETS_SKELETON AS
WITH base AS (
    SELECT SEQ4() AS n, UNIFORM(1, 500, RANDOM(21)) AS cust_n, UNIFORM(0, 19, RANDOM(22)) AS cat_n,
           UNIFORM(1, 100, RANDOM(23)) AS pri_roll, UNIFORM(1, 100, RANDOM(24)) AS status_roll,
           UNIFORM(0, 364, RANDOM(25)) AS age_days, UNIFORM(1, 100, RANDOM(26)) AS lang_roll,
           UNIFORM(0, 4, RANDOM(27)) AS tone_n
    FROM TABLE(GENERATOR(ROWCOUNT => 3000))
), cats AS (
    SELECT PRODUCT_ID, CATEGORY, ROW_NUMBER() OVER (ORDER BY PRODUCT_ID, CATEGORY) - 1 AS cat_n FROM TICKET_CATEGORIES
)
SELECT
    'T' || LPAD(b.n + 1, 5, '0')                             AS TICKET_ID,
    'C' || LPAD(b.cust_n, 4, '0')                            AS CUSTOMER_ID,
    c.PRODUCT_ID, c.CATEGORY,
    CASE WHEN b.pri_roll <= 10 THEN 'URGENT' WHEN b.pri_roll <= 35 THEN 'HIGH'
         WHEN b.pri_roll <= 75 THEN 'MEDIUM' ELSE 'LOW' END   AS PRIORITY,
    CASE WHEN b.age_days < 7 AND b.status_roll <= 60 THEN 'OPEN'
         WHEN b.status_roll <= 15 THEN 'OPEN'
         WHEN b.status_roll <= 25 THEN 'PENDING_CUSTOMER'
         ELSE 'RESOLVED' END                                  AS STATUS,
    DATEADD(hour, -b.age_days * 24 - UNIFORM(0, 23, RANDOM(28)), '2026-11-01 12:00:00'::TIMESTAMP_NTZ) AS CREATED_AT,
    CASE WHEN b.lang_roll <= 88 THEN 'English' WHEN b.lang_roll <= 92 THEN 'Spanish'
         WHEN b.lang_roll <= 96 THEN 'German' ELSE 'French' END AS LANGUAGE,
    ARRAY_CONSTRUCT('calm and precise','frustrated','terse, few details','very detailed and technical',
                    'polite but confused')[b.tone_n]::STRING  AS TONE
FROM base b JOIN cats c ON b.cat_n = c.cat_n;

CREATE OR REPLACE TABLE TICKETS AS
SELECT
    s.*,
    cu.TIER,
    p.PRODUCT_NAME,
    AI_COMPLETE(
        model => 'llama3.3-70b',
        prompt => 'Write one realistic customer support ticket for the SaaS product "' || p.PRODUCT_NAME ||
                  '" (' || p.DESCRIPTION || '). Issue category: ' || s.CATEGORY ||
                  '. Priority as perceived by the customer: ' || s.PRIORITY ||
                  '. Customer company: ' || cu.COMPANY_NAME || ' (' || cu.TIER || ' tier). Tone: ' || s.TONE ||
                  '. Write it in ' || s.LANGUAGE || '. Include at least one concrete detail such as an error code, ID, ' ||
                  'timestamp, amount, or browser/OS. Do not mention the category name verbatim. Subject under 12 words; body 2-6 sentences.',
        model_parameters => {'temperature': 0.9, 'max_tokens': 400},
        response_format => {'type': 'json', 'schema': {'type': 'object',
            'properties': {'subject': {'type': 'string'}, 'body': {'type': 'string'}},
            'required': ['subject', 'body']}}
    ) AS GEN
FROM TICKETS_SKELETON s
JOIN CUSTOMERS cu ON cu.CUSTOMER_ID = s.CUSTOMER_ID
JOIN PRODUCTS p   ON p.PRODUCT_ID   = s.PRODUCT_ID;

CREATE OR REPLACE TABLE TICKETS AS
SELECT TICKET_ID, CUSTOMER_ID, PRODUCT_ID, PRODUCT_NAME, CATEGORY, PRIORITY, STATUS, CREATED_AT,
       LANGUAGE, TIER,
       GEN:subject::STRING AS SUBJECT,
       GEN:body::STRING    AS BODY
FROM TICKETS;

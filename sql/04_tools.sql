ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:04_tools';
-- Retrieval tools for the support agent: Cortex Search (KB) and semantic view (ticket analytics).
-- Custom tools live in 05_custom_tools.sql.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

CREATE OR REPLACE CORTEX SEARCH SERVICE KB_SEARCH
    ON CONTENT
    ATTRIBUTES PRODUCT_NAME, CATEGORY, ARTICLE_TYPE, TITLE
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 day'
AS
SELECT ARTICLE_ID, PRODUCT_NAME, CATEGORY, ARTICLE_TYPE, TITLE, TITLE || '\n\n' || CONTENT AS CONTENT
FROM KB_ARTICLES;

CREATE OR REPLACE SEMANTIC VIEW SUPPORT_ANALYTICS
  TABLES (
    tickets AS CT_DEMO_DB.CORTEX_TRAINING.TICKETS
      PRIMARY KEY (TICKET_ID)
      COMMENT = 'One row per customer support ticket',
    customers AS CT_DEMO_DB.CORTEX_TRAINING.CUSTOMERS
      PRIMARY KEY (CUSTOMER_ID)
      COMMENT = 'One row per customer account',
    products AS CT_DEMO_DB.CORTEX_TRAINING.PRODUCTS
      PRIMARY KEY (PRODUCT_ID)
      COMMENT = 'Northwind Cloud products'
  )
  RELATIONSHIPS (
    tickets_to_customers AS tickets (CUSTOMER_ID) REFERENCES customers,
    tickets_to_products  AS tickets (PRODUCT_ID)  REFERENCES products
  )
  DIMENSIONS (
    tickets.ticket_id AS TICKET_ID COMMENT = 'Ticket identifier, e.g. T00042',
    tickets.category AS CATEGORY WITH SYNONYMS = ('issue type', 'problem type') COMMENT = 'Issue category such as sync_failure or invoice_dispute',
    tickets.priority AS PRIORITY COMMENT = 'URGENT, HIGH, MEDIUM, or LOW',
    tickets.status AS STATUS COMMENT = 'OPEN, PENDING_CUSTOMER, or RESOLVED',
    tickets.language AS LANGUAGE COMMENT = 'Language the ticket was written in',
    tickets.created_date AS DATE(CREATED_AT) COMMENT = 'Date the ticket was opened',
    tickets.created_week AS DATE_TRUNC('week', CREATED_AT) COMMENT = 'Week the ticket was opened',
    customers.customer_id AS CUSTOMER_ID COMMENT = 'Customer identifier, e.g. C0042',
    customers.company_name AS COMPANY_NAME WITH SYNONYMS = ('customer', 'account', 'company') COMMENT = 'Customer company name',
    customers.tier AS TIER WITH SYNONYMS = ('plan', 'support tier') COMMENT = 'ENTERPRISE, BUSINESS, or STARTER',
    customers.country AS COUNTRY COMMENT = 'Customer country code',
    products.product_name AS PRODUCT_NAME WITH SYNONYMS = ('product') COMMENT = 'Product the ticket is about'
  )
  METRICS (
    tickets.ticket_count AS COUNT(TICKET_ID) COMMENT = 'Number of tickets',
    tickets.open_ticket_count AS COUNT_IF(STATUS = 'OPEN') COMMENT = 'Number of open tickets',
    tickets.urgent_ticket_count AS COUNT_IF(PRIORITY = 'URGENT') COMMENT = 'Number of urgent tickets',
    customers.total_contract_value AS SUM(ANNUAL_CONTRACT_VALUE) COMMENT = 'Sum of annual contract value'
  )
  COMMENT = 'Support ticket analytics for Northwind Cloud';

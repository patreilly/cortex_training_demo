ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:03_kb_articles';
-- ~200 KB articles: 10 per category (troubleshooting, how-to, policy), plus policy docs.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

CREATE OR REPLACE TABLE KB_ARTICLES AS
WITH variants AS (
    SELECT SEQ4() AS v,
           ARRAY_CONSTRUCT('troubleshooting guide','how-to guide','known issue and workaround',
                           'FAQ','error code reference','configuration best practices',
                           'limits and quotas','release note describing a fix',
                           'admin guide','step-by-step recovery procedure')[SEQ4()]::STRING AS article_type
    FROM TABLE(GENERATOR(ROWCOUNT => 10))
), gen AS (
    SELECT c.PRODUCT_ID, p.PRODUCT_NAME, c.CATEGORY, v.article_type,
        AI_COMPLETE(
            model => 'llama3.3-70b',
            prompt => 'Write an internal knowledge base article for support agents at Northwind Cloud. Product: "' ||
                      p.PRODUCT_NAME || '" (' || p.DESCRIPTION || '). Topic area: ' || c.CATEGORY ||
                      '. Article type: ' || v.article_type || '. Use specific, plausible details: error codes of the form NW-#### ' ||
                      'unique to this article, setting names, numeric limits, and numbered resolution steps. 150-300 words.',
            model_parameters => {'temperature': 0.7, 'max_tokens': 700},
            response_format => {'type': 'json', 'schema': {'type': 'object',
                'properties': {'title': {'type': 'string'}, 'content': {'type': 'string'}},
                'required': ['title', 'content']}}
        ) AS g
    FROM TICKET_CATEGORIES c
    CROSS JOIN variants v
    JOIN PRODUCTS p ON p.PRODUCT_ID = c.PRODUCT_ID
)
SELECT 'KB' || LPAD(ROW_NUMBER() OVER (ORDER BY PRODUCT_ID, CATEGORY, article_type), 4, '0') AS ARTICLE_ID,
       PRODUCT_ID, PRODUCT_NAME, CATEGORY, article_type AS ARTICLE_TYPE,
       g:title::STRING AS TITLE, g:content::STRING AS CONTENT
FROM gen
WHERE g:content IS NOT NULL;

-- Support policy articles the agent must follow (refunds, escalation, SLAs).
INSERT INTO KB_ARTICLES
SELECT * FROM VALUES
 ('KBPOL1','ALL','All products','policy','policy','Refund policy',
  'Refunds are allowed only within 30 days of the invoice date. ENTERPRISE customers may receive refunds up to 100% of the disputed amount; BUSINESS up to 50%; STARTER customers are not eligible for refunds and should be offered account credit instead. Support agents must never promise a refund amount; they create a BILLING escalation and the billing team confirms.'),
 ('KBPOL2','ALL','All products','policy','policy','Engineering escalation policy',
  'Escalate to the ENGINEERING team only when (a) the customer tier allows engineering escalation per SLA policy and (b) the issue is a reproducible product defect or outage, not a configuration or how-to question. All other escalations go to TIER2_SUPPORT. Security concerns (suspected account compromise, leaked API keys) always go to SECURITY regardless of tier.'),
 ('KBPOL3','ALL','All products','policy','policy','SLA and priority policy',
  'First-response and resolution targets depend on customer tier: ENTERPRISE 1h/24h, BUSINESS 4h/72h, STARTER 24h/168h. URGENT priority is reserved for full outages or security incidents affecting production. Agents should downgrade priority when the customer-stated urgency does not match these criteria and explain why.');

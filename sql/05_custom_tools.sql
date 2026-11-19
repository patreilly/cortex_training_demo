ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:05_custom_tools';
-- Custom tools for the support agent: ticket lookup, customer lookup, escalation.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

CREATE OR REPLACE FUNCTION GET_TICKET(P_TICKET_ID STRING)
RETURNS OBJECT
AS
$$
    SELECT OBJECT_CONSTRUCT('ticket_id', t.TICKET_ID, 'customer_id', t.CUSTOMER_ID, 'product', t.PRODUCT_NAME,
                            'category', t.CATEGORY, 'priority', t.PRIORITY, 'status', t.STATUS,
                            'created_at', t.CREATED_AT, 'language', t.LANGUAGE,
                            'subject', t.SUBJECT, 'body', t.BODY)
    FROM CT_DEMO_DB.CORTEX_TRAINING.TICKETS t
    WHERE t.TICKET_ID = UPPER(TRIM(P_TICKET_ID))
$$;

CREATE OR REPLACE FUNCTION LOOKUP_CUSTOMER(P_CUSTOMER_ID STRING)
RETURNS OBJECT
AS
$$
    SELECT OBJECT_CONSTRUCT('customer_id', c.CUSTOMER_ID, 'company_name', c.COMPANY_NAME, 'tier', c.TIER,
                            'country', c.COUNTRY, 'customer_since', c.CUSTOMER_SINCE,
                            'annual_contract_value', c.ANNUAL_CONTRACT_VALUE, 'csm_email', c.CSM_EMAIL,
                            'sla_first_response_hours', s.FIRST_RESPONSE_HOURS,
                            'sla_resolution_hours', s.RESOLUTION_HOURS,
                            'can_escalate_to_engineering', s.CAN_ESCALATE_TO_ENG,
                            'open_tickets', COUNT_IF(t.STATUS = 'OPEN'))
    FROM CT_DEMO_DB.CORTEX_TRAINING.CUSTOMERS c
    JOIN CT_DEMO_DB.CORTEX_TRAINING.SLA_POLICIES s ON s.TIER = c.TIER
    LEFT JOIN CT_DEMO_DB.CORTEX_TRAINING.TICKETS t ON t.CUSTOMER_ID = c.CUSTOMER_ID
    WHERE c.CUSTOMER_ID = UPPER(TRIM(P_CUSTOMER_ID))
    GROUP BY c.CUSTOMER_ID, c.COMPANY_NAME, c.TIER, c.COUNTRY, c.CUSTOMER_SINCE, c.ANNUAL_CONTRACT_VALUE,
             c.CSM_EMAIL, s.FIRST_RESPONSE_HOURS, s.RESOLUTION_HOURS, s.CAN_ESCALATE_TO_ENG
$$;

CREATE OR REPLACE PROCEDURE CREATE_ESCALATION(P_TICKET_ID STRING, P_TEAM STRING, P_REASON STRING)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    IF (UPPER(:P_TEAM) NOT IN ('ENGINEERING', 'TIER2_SUPPORT', 'BILLING', 'SECURITY')) THEN
        RETURN 'ERROR: team must be one of ENGINEERING, TIER2_SUPPORT, BILLING, SECURITY';
    END IF;
    IF (NOT EXISTS (SELECT 1 FROM CT_DEMO_DB.CORTEX_TRAINING.TICKETS WHERE TICKET_ID = UPPER(TRIM(:P_TICKET_ID)))) THEN
        RETURN 'ERROR: ticket ' || :P_TICKET_ID || ' not found';
    END IF;
    INSERT INTO CT_DEMO_DB.CORTEX_TRAINING.ESCALATIONS (TICKET_ID, TEAM, REASON)
        VALUES (UPPER(TRIM(:P_TICKET_ID)), UPPER(:P_TEAM), :P_REASON);
    RETURN 'Escalation created for ' || UPPER(TRIM(:P_TICKET_ID)) || ' to ' || UPPER(:P_TEAM);
END;
$$;

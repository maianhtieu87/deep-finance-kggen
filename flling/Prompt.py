class SystemSetUp:
    def __init__(self):
        self.summary_set_up = '''You are a professional financial analysis chatbot, you help investors read and get insights from financial documents'''


class FilingSummary:
    def __init__(self, content, filedAt):
        self.filedAt = filedAt  # Corrected typo from fileAt
        self.content = content
       
        self.f10k_item8 = f'''
        Form 10-K Item 8 filed on: {filedAt}


        Summarize the following Form 10-K Item 8 (Financial Statements and Supplementary Data) for a financial investor. Follow these steps:
        1. Extract and summarize key financial tables:
           - Balance Sheet: Total assets, total liabilities, shareholders' equity.
           - Income Statement: Revenue, gross profit, operating income, net income.
           - Cash Flow Statement: Operating cash flow, investing cash flow, financing cash flow.
           - Statement of Shareholders' Equity: Changes in equity components.
           Provide quantitative metrics for the reported period and, if available, comparisons with prior periods.
        2. Calculate and present key financial ratios based on the data, such as:
           - Current ratio (current assets / current liabilities)
           - Debt-to-equity ratio (total debt / total equity)
           - Gross margin (gross profit / revenue)
           - Net profit margin (net income / revenue)
           - Return on equity (net income / average shareholders' equity)
        3. Summarize the auditor's report, noting if it is unqualified, qualified, or if there are any going concern issues.
        4. Highlight any significant changes or trends in the financial data, such as substantial increases in debt, declines in revenue, or shifts in cash flow patterns.
        5. Summarize critical disclosures from the notes to the financial statements, including:
           - Accounting policies
           - Contingencies and legal proceedings
           - Related party transactions
           - Subsequent events
        6. Ensure the summary is concise and includes only information explicitly stated in the content. Avoid speculation or assumptions not supported by the text.


        If the content is incomplete or a placeholder, respond with: "The provided Item 8 content is insufficient. Please provide the full text or tables from the 10-K filing for analysis."


        Article: {content}
        '''


        self.f10k_item1A = f'''
        Form 10-K Item 1A filed on: {filedAt}


        Summarize the following Form 10-K Item 1A (Risk Factors) for a financial investor. Follow these steps:
        1. Identify and list the main categories of risks discussed (e.g., macroeconomic, operational, financial, regulatory).
        2. Highlight the most significant risks that could materially affect the company's business, financial condition, or results of operations.
        3. Note any new risks or significant changes in risks compared to previous filings, if mentioned.
        4. Summarize how these risks could impact the company's performance or stock price, based on the descriptions provided.
        5. Ensure the summary is concise and includes only information explicitly stated in the content. Avoid speculation or assumptions not supported by the text.


        If the content is incomplete or a placeholder, respond with: "The provided Item 1A content is insufficient. Please provide the full text from the 10-K filing for analysis."


        Article: {content}
        '''


        self.f10k_item7 = f'''
        Form 10-K Item 7 filed on: {filedAt}


        Summarize the following Form 10-K Item 7 (Management's Discussion and Analysis of Financial Condition and Results of Operations) for a financial investor. Follow these steps:
        1. Summarize the key factors that drove the company's financial performance during the reported period, including significant changes in revenue, expenses, and profitability.
        2. Highlight management's discussion of liquidity and capital resources, including cash flow trends and funding sources.
        3. Identify any forward-looking statements or guidance provided by management regarding future performance or expectations.
        4. Note any material trends, uncertainties, or risks that management has identified that could affect future results.
        5. Ensure the summary is concise and includes only information explicitly stated in the content. Avoid speculation or assumptions not supported by the text.


        If the content is incomplete or a placeholder, respond with: "The provided Item 7 content is insufficient. Please provide the full text from the 10-K filing for analysis."


        Article: {content}
        '''


        self.f10q_item1A = f'''
        Form 10-Q Item 1A filed on: {filedAt}


        Summarize the following Form 10-Q Item 1A (Risk Factors) for a financial investor. Follow these steps:
        1. Determine if the company has disclosed any new or updated risk factors compared to their most recent 10-K filing.
        2. If there are new or updated risks, summarize the key points of these risks and their potential impact on the company.
        3. If there are no material changes, note that the company has stated there are no material changes to the risk factors previously disclosed.
        4. Ensure the summary is concise and includes only information explicitly stated in the content. Avoid speculation or assumptions not supported by the text.


        If the content is incomplete or a placeholder, respond with: "The provided Item 1A content from the 10-Q is insufficient. Please provide the full text for analysis."


        Article: {content}
        '''


        self.f10q_item2 = f'''
        Form 10-Q Item 2 filed on: {filedAt}

        Summarize the following Form 10-Q Item 2 (Management's Discussion and Analysis of Financial Condition and Results of Operations) for a financial investor. Follow these steps:
        1. Summarize the key factors that drove the company's financial performance during the quarter, including significant changes in revenue, expenses, and profitability.
        2. Highlight management's discussion of liquidity and capital resources, including cash flow trends and funding sources.
        3. Identify any forward-looking statements or guidance provided by management regarding future performance or expectations.
        4. Note any material trends, uncertainties, or risks that management has identified that could affect future results.
        5. Ensure the summary is concise and includes only information explicitly stated in the content. Avoid speculation or assumptions not supported by the text.

        If the content is incomplete or a placeholder, respond with: "The provided Item 2 content from the 10-Q is insufficient. Please provide the full text for analysis."

        Article: {content}
        '''


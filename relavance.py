from llama_cpp import Llama, LlamaGrammar
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('C:\\Users\\rohil\\Desktop\\New folder (5)\\reuters.db')
cursor = conn.cursor()

# Add a new column 'relevance' to store the final score, if it doesn't already exist.
try:
    cursor.execute("ALTER TABLE news ADD COLUMN relevance INTEGER")
except sqlite3.OperationalError:
    pass  # Ignore if the column already exists.

# Fetch only news articles where the relevance score is NULL (i.e., they haven't been processed yet)
cursor.execute("SELECT headline, text, rowid FROM news WHERE relevance IS NULL")
news_articles = cursor.fetchall()

# Prompt template
prompt_template = """
News Article:
Headline: {headline}

Text: {text}

Based on the above news article, provide a yes/no - answer for each of these points.

Topic: Financial Market Relevance
1. Does the news affect major stock indices (e.g., S&P 500, FTSE 100, Nikkei 225)?
2. Does the news involve a major global economic policy change (e.g., interest rates, trade policies)?
3. Is the news about a significant geopolitical event (e.g., war, major diplomatic agreements)?
4. Does the news impact major global currencies (e.g., USD, EUR, JPY)?
5. Is the news related to significant changes in commodity prices (e.g., oil, gold)?
6. Does the news concern major corporations or industries (e.g., tech giants, automotive industry)?
7. Is there an impact on major global financial institutions (e.g., banks, hedge funds)?
8. Does the news affect international trade or supply chains?
9. Is the news related to significant economic data releases (e.g., GDP, unemployment rates)?
10. Does the news lead to significant market speculation or volatility?

Respond in this way:
    - Provide your reasoning for each question.
    - THEN provide a final answer for the yes/no question.
"""

root_rule = 'root ::= ' + ' '.join([f'"Reasoning {i} (1 sentence max): " reasoning "\\n" "Answer {i} (yes/no): " yesno "\\n"' for i in range(1, 11)])
grammar = f"""
{root_rule}
reasoning ::= sentence
sentence ::= [^.!?]+[.!?]
yesno ::= "yes" | "no"
"""

grammar = LlamaGrammar.from_string(grammar)

# Model path - change this to the correct path for your model
llm = Llama(
    model_path = r"D:/SENTI/Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf",
    n_gpu_layers = -1,
    seed = 123,
    n_ctx = 8192,
    verbose = False
)

for headline, text, rowid in news_articles:
    print(f"Processing article with rowid {rowid}...")  # Print operation message

    try:
        prompt = prompt_template.format(headline=headline, text=text)
        
        output = llm(
            prompt,
            max_tokens=2000,
            stop=["</s>"],
            echo=False,
            grammar=grammar
        )
        
        response = output["choices"][0]["text"].strip()
        yes_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "yes")
        no_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "no")
        
        print(f"Headline: {headline}")
        print(f"Response:\n{response}")
        print(f"Yes answers: {yes_count}")
        print(f"No answers: {no_count}")
        print(f"Final Score: {yes_count}")
        print("-" * 50)

        # Update the 'relevance' column with the final score (yes_count) for the current article
        cursor.execute("UPDATE news SET relevance = ? WHERE rowid = ?", (yes_count, rowid))
        
    except Exception as e:
        # If an error occurs, set relevance to NULL and continue with the next article
        cursor.execute("UPDATE news SET relevance = NULL WHERE rowid = ?", (rowid,))
        print(f"Error processing article with rowid {rowid}: {e}")
    
    # Commit after every update
    conn.commit()

    print(f"Finished processing article with rowid {rowid}.")  # Print completion message

# Close the database connection
conn.close()

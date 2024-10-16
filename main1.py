from colorama import init, Fore, Style
import sqlite3
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time

# Initialize colorama
init(autoreset=True)

# Initialize GPT-Neo model from Hugging Face
def init_gpt_neo():
    print(Fore.YELLOW + "Initializing GPT-Neo model..." + Style.RESET_ALL)
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    return generator

# Create or connect to the SQLite database
def create_database():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 question TEXT NOT NULL, 
                 answer TEXT NOT NULL, 
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Retrieve all interactions from the database
def get_all_interactions():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interactions")
    interactions = c.fetchall()
    conn.close()
    return interactions

# Add a new question-answer interaction to the database
def add_interaction(question, answer):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# Display stored interactions
def view_interactions():
    interactions = get_all_interactions()

    if len(interactions) == 0:
        print(Fore.RED + "No interactions found." + Style.RESET_ALL)
    else:
        print(Fore.MAGENTA + "\nQuestion-Answer Interactions:" + Style.RESET_ALL)
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
        for interaction in interactions:
            print(f"ID: {interaction[0]}")
            print(f"Question: {interaction[1]}")
            print(f"Answer: {interaction[2]}")
            print(f"Timestamp: {interaction[3]}")
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

# Display intro banner
def display_intro():
    print(Fore.CYAN + """
 __      __                      _____________________________
/  \    /  \___________  _____  /  _____/\______   \__    ___/
\   \/\/   /  _ \_  __ \/     \/   \  ___ |     ___/ |    |
 \        (  <_> )  | \/  Y Y  \    \_\  \|    |     |    |
  \__/\  / \____/|__|  |__|_|  /\______  /|____|     |____|
       \/                    \/        \/

Welcome to WormGPT with GPT-Neo! Let's chat!
    """ + Style.RESET_ALL)

# Get the answer using GPT-Neo model
def get_gpt_neo_answer(generator, question):
    try:
        # Generate the response
        response = generator(question, max_length=150, do_sample=True)
        answer = response[0]['generated_text'].strip()
        return answer
    except Exception as e:
        print(Fore.RED + f"Error generating answer: {str(e)}" + Style.RESET_ALL)
        return None

# Analyze patterns in past interactions
def analyze_interactions():
    interactions = get_all_interactions()

    if len(interactions) == 0:
        print(Fore.RED + "No interactions found for analysis." + Style.RESET_ALL)
        return

    df = pd.DataFrame(interactions, columns=['id', 'question', 'answer', 'timestamp'])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['question'] + " " + df['answer'])

    num_topics = 5
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    print(Fore.MAGENTA + "\nDiscovered Patterns:" + Style.RESET_ALL)
    print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-10:]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Pattern {topic_idx + 1}: {', '.join(top_words)}")
    print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

# Main interaction loop
def main():
    # Initialize GPT-Neo
    generator = init_gpt_neo()
    
    # Create database if not exists
    create_database()
    
    # Display intro
    display_intro()

    history = []
    last_question = None
    last_answer = None

    while True:
        question = input(Fore.CYAN + "Enter your question: " + Style.RESET_ALL)

        if question.lower() == 'exit':
            print(Fore.YELLOW + "WormGPT welcomes you - See you soon." + Style.RESET_ALL)
            break
        elif question.lower() == 'history':
            print(Fore.MAGENTA + "\nQuestion History:" + Style.RESET_ALL)
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
            for idx, item in enumerate(history, start=1):
                print(f"{idx}. Question: {item['question']}")
                print(f"   Answer: {item['answer']}")
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
            continue
        elif question.lower() == 'clear':
            history.clear()
            print(Fore.GREEN + "\nQuestion history cleared." + Style.RESET_ALL)
            continue
        elif question.lower() == 'repeat':
            if last_question and last_answer:
                print(Fore.YELLOW + "\nLast Question:" + Style.RESET_ALL)
                print(last_question)
                print(Fore.YELLOW + "Last Answer:" + Style.RESET_ALL)
                print(last_answer)
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
            else:
                print(Fore.RED + "\nNo last question and answer available." + Style.RESET_ALL)
            continue
        elif question.lower() == 'view':
            view_interactions()
            continue
        elif question.lower() == 'analyze':
            analyze_interactions()
            continue

        if question:
            # Get answer from GPT-Neo
            answer = get_gpt_neo_answer(generator, question)
            if answer:
                print(Fore.GREEN + "\nAnswer:" + Style.RESET_ALL)
                print(Fore.MAGENTA + answer + Style.RESET_ALL)
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

                # Add interaction to the database
                add_interaction(question, answer)
                
                # Store interaction in history
                history.append({"question": question, "answer": answer})
                last_question = question
                last_answer = answer
        else:
            print(Fore.RED + "Please enter a question." + Style.RESET_ALL)


if __name__ == '__main__':
    main()
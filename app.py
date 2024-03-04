import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
import datetime 

# API Keys & Pinecone Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENVIRONMENT = os.getenv("PINECONE_API_ENVIRONMENT")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)
PINECONE_INDEX_NAME = 'chat'

# Streamlit Setup 
st.set_page_config(
    page_title="Ramayan GPT", 
    page_icon="🤖",
    layout="centered" 
)


st.title('Ramayan GPT')

st.write("If you could seek guidance from the wisdom of the Ramayana, what would you ask?")
# st.markdown('\n')
st.write('''\n\n Here's some examples of what you can ask:
1. Ravana, despite his power, fell due to his choices. How can I ensure my actions are guided by righteousness, even when faced with temptation?
2. Hanuman's devotion was a source of immense power. How do I find a similar source of inspiration to overcome my own challenges?.
3. The love between Rama and Sita endured through great trials. How can I nurture compassion and understanding within my own relationships??
''')

# Langchain Models & Setup
llm = ChatOpenAI(temperature=0.9, max_tokens=150, model='gpt-3.5-turbo-0613')
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
chain = load_qa_chain(llm)

# Initialize chat history (if not in session state)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to display chat history 
def display_chat_history(): 
    

    for message in st.session_state['chat_history']:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        role = "You" if message['role'] == 'user' else "Valmiki Says" 

        if role == "You":
            color = "#000000"  
        else:
            color = "#000000" 

        st.write(f"<p style='color:{color};'> <b>{role} ({timestamp}):</b> {message['text']}</p>", unsafe_allow_html=True)

# Input & Response Area
with st.container(): 
    usr_input = st.text_input('How are you feeling? Ask a question or describe your situation below, and then press Enter.')

    if usr_input:
        with st.spinner("Thinking..."):
            try:
                st.session_state['chat_history'].append({'role': 'user', 'text': usr_input})
                search = docsearch.similarity_search(usr_input)
                response = chain.run(input_documents=search, question=usr_input) 
                st.session_state['chat_history'].append({'role': 'bot', 'text': response})

            except pinecone.ApiException as e: 
                st.error("Pinecone Error: Check your connection and index.")
                print(e)

            except langchain.OpenAIError as e:  
                st.error("Chatbot Error: Something went wrong with the AI model.")
                print(e)

            except Exception as e: 
                st.error("An unexpected error occurred. Please try again.")
                print(e)



# Display chat history
display_chat_history() 


st.write('\n\n\n\n\n\n\n')

st.write('''Note: This is an AI model trained on The Ramayan of Valmiki and it generates responses from that perspective.''')

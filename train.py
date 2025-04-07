import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import difflib
import joblib

# Sample locations data
locations = ["Delhi", "Agra", "Kolkata", "Thiruvananthapuram", "Calicut", "Hyderabad", 
             "Kochi", "Ahmedabad", "Punjab", "Maharashtra", "Mumbai", "Jaipur", 
             "Rajasthan", "Bhopal", "Haryana", "Tamil Nadu", "Uttar Pradesh", 
             "Lucknow", "Pune", "Madhya Pradesh", "Patna", "Bihar", "Goa", "Gujarat", 
             "Haryana", "Karnataka", "Bangalore", "Kerala", "Maharashtra"]

# Create a DataFrame for the locations
df_locations = pd.DataFrame(locations, columns=['correct_name'])

# Generate synthetic variations by removing spaces, adding common misspellings, and extra spaces
def generate_variations(name):
    variations = [name.replace(' ', '').lower(), name.lower(), name.replace(' ', '  ').lower()]
    return variations

# Create a new DataFrame with variations
data = {'correct_name': [], 'variation': []}
for name in df_locations['correct_name']:
    for variation in generate_variations(name):
        data['correct_name'].append(name)
        data['variation'].append(variation)

df_variations = pd.DataFrame(data)

# Preprocess text: lowercase and remove extra spaces
def preprocess(text):
    return ' '.join(text.lower().split())

df_variations['correct_name'] = df_variations['correct_name'].apply(preprocess)
df_variations['variation'] = df_variations['variation'].apply(preprocess)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_variations, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer().fit(train_df['correct_name'])
train_vectors = vectorizer.transform(train_df['correct_name'])
test_vectors = vectorizer.transform(test_df['variation'])

# Extract train names for use in the search function
train_names = train_df['correct_name'].values

# Function to find the best match using cosine similarity and difflib
def find_best_match(query, train_vectors, train_names):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, train_vectors)
    best_match_index = similarities.argmax()
    best_match_name = train_names[best_match_index]
    
    # Use difflib to refine the best match
    difflib_scores = [difflib.SequenceMatcher(None, query, name).ratio() for name in train_names]
    best_difflib_match_index = difflib_scores.index(max(difflib_scores))
    best_difflib_match_name = train_names[best_difflib_match_index]
    
    # Combine scores
    combined_scores = [(similarity + difflib_score) / 2 for similarity, difflib_score in zip(similarities[0], difflib_scores)]
    best_combined_match_index = combined_scores.index(max(combined_scores))
    best_combined_match_name = train_names[best_combined_match_index]
    
    return best_combined_match_name

# Function to save the model
def save_model():
    joblib.dump(vectorizer, 'location_vectorizer.pkl')
    joblib.dump(train_names, 'location_train_names.pkl')
    joblib.dump(train_vectors, 'location_train_vectors.pkl')

# Function to load the model
def load_model():
    vectorizer = joblib.load('location_vectorizer.pkl')
    train_names = joblib.load('location_train_names.pkl')
    train_vectors = joblib.load('location_train_vectors.pkl')
    return vectorizer, train_names, train_vectors

# Save the model
save_model()

# Test the model
correct_predictions = 0
for i, row in test_df.iterrows():
    best_match = find_best_match(row['variation'], train_vectors, train_names)
    if best_match == row['correct_name']:
        correct_predictions += 1

accuracy = correct_predictions / len(test_df)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Example usage with a sample query
query = "Kolkata"
best_match = find_best_match(query, train_vectors, train_names)
print(f"Best match for '{query}': {best_match}")
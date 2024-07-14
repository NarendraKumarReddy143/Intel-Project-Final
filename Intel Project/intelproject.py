import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path, chunksize=10000):
    """
    Load the dataset from a CSV file in chunks.
    """
    try:
        # Read the first chunk to initialize the DataFrame
        df_iter = pd.read_csv(file_path, chunksize=chunksize)
        df = next(df_iter)
        
        # Concatenate remaining chunks
        for chunk in df_iter:
            df = pd.concat([df, chunk], ignore_index=True)
        
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def downcast_dtypes(df):
    """
    Downcast numerical columns to reduce memory usage.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing duplicates.
    """
    # Display the number of missing values before cleaning
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    print("\n")

    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float32', 'float64', 'int32', 'int64']:
            # Fill numerical columns with the mean
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            # Fill categorical columns with the mode
            df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Display the number of missing values after cleaning
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    print("\n")

    return df

def identify_target_column(df):
    """
    Identify the target column automatically.
    """
    # Heuristic: Assume the target column is the non-numerical column with the fewest unique values
    non_numerical_columns = df.select_dtypes(include=['object', 'category']).columns
    target_column = None
    min_unique_values = float('inf')
    
    for column in non_numerical_columns:
        unique_values = df[column].nunique()
        if unique_values < min_unique_values:
            min_unique_values = unique_values
            target_column = column
    
    if target_column is None:
        raise ValueError("No suitable target column found.")
    
    return target_column

def generate_insights(df, sample_size=10000):
    """
    Generate insights and visualizations from the dataset.
    """
    if df is not None:
        print("Generating insights...")
        
        # Clean the data
        df = clean_data(df)

        # Downcast dtypes to save memory
        df = downcast_dtypes(df)

        # Sample the data if it's too large
        if len(df) > sample_size:
            df = df.sample(sample_size)
            print(f"Data sampled to {sample_size} rows for analysis.")

        # Display basic information about the dataset
        print("Dataset Information:")
        print(df.info())
        print("\n")

        # Display the first few rows of the dataset
        print("First few rows of the dataset:")
        print(df.head())
        print("\n")

        # Display summary statistics for numerical columns only
        print("Summary statistics (Numerical Columns):")
        numerical_df = df.select_dtypes(include=['float32', 'int32'])
        print(numerical_df.describe())
        print("\n")

        # Generate and display correlation matrix for numerical columns
        print("Correlation matrix (Numerical Columns):")
        corr_matrix = numerical_df.corr()
        print(corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        # Identify and visualize trends and patterns for numerical columns
        for column in numerical_df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=numerical_df, x=numerical_df.index, y=column)
            plt.title(f'Trend for {column}')
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.show()

        # Pairplot for numerical columns to identify relationships
        if len(numerical_df.columns) > 1:
            sns.pairplot(numerical_df)
            plt.suptitle('Pairplot of Numerical Columns')
            plt.show()

        # Analyze and visualize categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=column)
            plt.title(f'Count plot for {column}')
            plt.xticks(rotation=45)
            plt.show()

            # If there are numerical columns, create box plots to show distribution
            if len(numerical_df.columns) > 0:
                for num_col in numerical_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, x=column, y=num_col)
                    plt.title(f'Box plot of {num_col} by {column}')
                    plt.xticks(rotation=45)
                    plt.show()

    else:
        print("No dataset to generate insights from.")

def perform_gradient_boosting(df):
    """
    Perform Gradient Boosting using HistGradientBoostingClassifier.
    """
    if df is None:
        print("No dataset to perform gradient boosting.")
        return

    target_column = identify_target_column(df)
    print(f"Identified target column: {target_column}")

    # Encode categorical columns
    df_encoded = df.copy()
    for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
    
    # Prepare the data for training
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    file_path = 'iris.csv'  # Replace with your dataset file path
    df = load_dataset(file_path)
    if df is not None:
        generate_insights(df)
        perform_gradient_boosting(df)
    else:
        print("Failed to load the dataset.")

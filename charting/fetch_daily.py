# import pandas as pd

# def reconstruct_structure(input_csv):
#     # Read the input CSV file
#     df = pd.read_csv(input_csv)

#     # Convert date to the desired format (e.g., from '2/4/2024' to '2024-02-04')
#     df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

#     # Remove the 'Symbol' column
#     df = df.drop(columns=['Symbol', 'Percent Change'])

#     # Remove commas from the 'Volume' column
#     df['Volume'] = df['Volume'].str.replace(',', '')

#     df = df[::-1].reset_index(drop=True)

#     df.insert(0, 'Index', range(1136, 1136 + len(df)))

#     # Save the reconstructed DataFrame to a new CSV file
#     output_csv = 'output.csv'
#     df.to_csv(output_csv, index=False)
#     print(f"Output CSV file '{output_csv}' generated successfully.")

# # Example usage
# input_csv = 'df2.csv'  # Update with the actual path to your input CSV file

# reconstruct_structure(input_csv)

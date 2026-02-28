# GDS-Eksamen-FakeNews
Vores eksamens projekt

Data_Processing can be called using:    python Data_Processing <PathToCSV> <optional row limit>

Example usage:    python Data_Processing <Data\news_sample.csv>
Example usage:    python Data_Processing <Data\995,000_rows.csv> 1000




Notes for report: Currently everything is saved in memory which is very bad. Personally I saw 20GB of memory in use by Python when running the 995k file.
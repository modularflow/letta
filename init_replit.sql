-- Title: Init Letta Database for Replit

-- Connect to the default database
\c postgres

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for letta
CREATE SCHEMA IF NOT EXISTS letta;

-- Set the search path
SET search_path TO letta;

-- Grant necessary permissions
GRANT ALL ON SCHEMA letta TO public;
GRANT ALL ON ALL TABLES IN SCHEMA letta TO public;
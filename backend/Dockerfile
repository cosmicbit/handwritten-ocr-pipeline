FROM postgres:16

# Optional: set defaults (can also be done in compose)
ENV POSTGRES_DB=backend_db
ENV POSTGRES_USER=backend_user
ENV POSTGRES_PASSWORD=supersecret123

# Optional: init SQL scripts
# Any .sql or .sh here runs on FIRST startup only
COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432

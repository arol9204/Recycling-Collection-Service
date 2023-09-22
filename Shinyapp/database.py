import psycopg2
#from psycopg2 import Error

# This is the pyhton guide notebook website
# https://pynative.com/python-postgresql-tutorial/

connection = psycopg2.connect(user="postgres",
                                  password="sqladmin",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    
    # This is neccesary to perform operations inside the database
cursor = connection.cursor()

    # # SQL query to CREATE a new table ---------------------------------------------------------------------------
    # create_table_query = '''CREATE TABLE mobile
    #       (ID INT PRIMARY KEY     NOT NULL,
    #       MODEL           TEXT    NOT NULL,
    #       PRICE         REAL); '''
    
    # # Execute a command: this creates a new table
    # cursor.execute(create_table_query)
    # connection.commit()
    # print("Table created successfully in PostgreSQL ")
    
    # Executing a SQL query to INSERT data into table ----------------------------------------------------------

    # There are two ways to INSERT values into a table

# 1. Inserting values by defining the values in the query
insert_query = """ INSERT INTO mobile (ID, MODEL, PRICE) VALUES (2, 'Iphone13', 1300) """
cursor.execute(insert_query)

    # # 2. Inserting values by predefining the values in a variable
    # # We used a parameterized query to use Python variables as parameter values at execution time. Using a parameterized query, we can pass python variables as a query parameter using placeholders (%s).
    # insert_query = """ INSERT INTO mobile (ID, MODEL, PRICE) VALUES (%s, %s, %s) """
    # id = 5
    # phone = "GooglePixel"
    # price = 700
    # record_to_insert = (id, phone, price)
    # cursor.execute(insert_query, record_to_insert)
    
connection.commit()
print("1 Record inserted succesfully")
    
    # # Executing a SQL query to update table --------------------------------------------------------------------
    # update_query = """ Update mobile set price = 1500 where id = 1"""
    # cursor.execute(update_query)
    # connection.commit()
    # count = cursor.rowcount
    # print(count, "Record updated succesfully")

# # Executing a SQL query to delete rows from a table -----------------------------------------------------------
# delete_query = """ Delete from mobile where id = 1"""
# cursor.execute(delete_query)
# connection.commit()
# count = cursor.rowcount
# print(count, "Record deleted successfully")
    
# Fetch result -------------------------------------------------------------
cursor.execute("SELECT * FROM mobile")
record = cursor.fetchall()
print("Result ", record)


# except (Exception, Error) as error:
#     print("Error while connecting to PostgreSQL", error)
# finally:
#    if connection:
cursor.close()
connection.close()
print("PostgreSQL connection is closed")

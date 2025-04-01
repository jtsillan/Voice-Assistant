import mysql.connector

def search_for_contacts():        
        connection = None
        cursor = None
        
        try:
            connection = mysql.connector.connect(user='root', host='127.0.0.1', database='thesis')
            cursor = connection.cursor(prepared=True, dictionary=True)
            query = "SELECT * FROM contacts AS c INNER JOIN users AS u ON c.users_id = u.id INNER JOIN phone_numbers AS pn ON c.phone_numbers_id = pn.id WHERE c.users_id = 1"
            cursor.execute(query)
            contacts = cursor.fetchall()
            
            if contacts == []:
                # Query didn't find anything --> 
                print(f"contacts: {type(contacts)}")
            
            #if contacts is not None: # or empty list/dictionary
            for index, contact in enumerate(contacts):
                print(f"contact {index}: {contact}")
                print(contact['id'], contact['contact_name'], contact['phone_number'])
                    
            # TODO: set contact['id'], contact['contact_name'] and contact['phone_number'] to self.contacts
            
        except Exception as ex:
            print(ex)

        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None and connection.is_connected():
                connection.close()
                
                
if __name__ == "__main__":
    search_for_contacts()
    
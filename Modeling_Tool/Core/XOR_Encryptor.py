import base64
import random
import pandas as pd
import numpy as np

class TextEncryptor:
    """
    Text encryption and decryption utility class based on the XOR algorithm.

    The class provides text encryption and decryption, and supports both single strings and entire pandas DataFrames.
    Encrypted data is encoded as URL-safe Base64, which makes it easy to store and transmit.

    Attributes:
        key (str): Key used for encryption and decryption. If None, an empty string is used as the key.
        suffix (str): Suffix appended to DataFrame column names after encryption. Defaults to '_encrypted'.

    Example:
        >>> encryptor = TextEncryptor(key="my_secret_key")
        >>> encrypted = encryptor.encrypt("Hello World")
        >>> decrypted = encryptor.decrypt(encrypted)
        >>> print(decrypted)  # Output: Hello World
    """

    def __init__(self, key=None, suffix='_encrypted'):
        """
        Initialize the encryptor instance.

        Parameters:
            key (str, optional): Key used for encryption and decryption. If None, an empty string is used as the key.
                               Note: data encrypted with an empty key is not confidential.
            suffix (str, optional): Suffix appended to column names when a DataFrame is encrypted.
                                  Defaults to '_encrypted'. The suffix is removed on decryption.
        """
        self.key = key
        self.suffix = suffix

    def encrypt(self, text):
        """
        Encrypt the input text.

        XOR the plaintext with the key, then output the result as URL-safe Base64.
        The encrypted result carries the original text length (the first 2 bytes), which is used for validation during decryption.

        Parameters:
            text (str): Plaintext string to encrypt.

        Returns:
            str: Encrypted string, encoded as URL-safe Base64.

        Raises:
            AttributeError: If the key attribute is None (when self.key is None, an empty string is actually used).

        Example:
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted = encryptor.encrypt("Hello")
            >>> print(encrypted)  # Output: AAU7AA8eCg==
        """
        # Text to bytes
        text_bytes = text.encode('utf-8')

        # Expand byte length
        key_bytes = (self.key * (len(text_bytes) // len(self.key) + 1)).encode('utf-8')
        key_bytes = key_bytes[:len(text_bytes)]

        # XOR encryption
        encrypted_bytes = bytes([text_bytes[i] ^ key_bytes[i] for i in range(len(text_bytes))])

        # Add 2 more bytes for verification
        length_byte = len(text).to_bytes(2, 'big')

        # combine
        result_bytes = length_byte + encrypted_bytes
        return base64.urlsafe_b64encode(result_bytes).decode('utf-8')

    def decrypt(self, encrypted_text):
        """
        Decrypt previously encrypted text.

        Decode the Base64 string, extract the length information (the first 2 bytes), then XOR the remaining bytes with the key to recover the plaintext.
        After decryption, the length of the recovered text is checked against the stored length to ensure data integrity.

        Parameters:
            encrypted_text (str): Base64-encoded string produced by the encrypt method.

        Returns:
            str: Original plaintext string after decryption.

        Raises:
            ValueError: If decryption fails. Possible causes include:
                       - Base64 decoding failed (the input is not a valid Base64 string)
                       - Length validation failed (the data was tampered with or a different key was used)
                       - Other decoding errors

        Example:
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted = encryptor.encrypt("Hello")
            >>> decrypted = encryptor.decrypt(encrypted)
            >>> print(decrypted)  # Output: Hello
        """
        try:
            # b64 decryption
            decoded_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))

            # extract length info
            text_length = int.from_bytes(decoded_bytes[:2], 'big')

            # extract encryption info
            encrypted_bytes = decoded_bytes[2:]

            # regenerate key
            key_bytes = (self.key * (len(encrypted_bytes) // len(self.key) + 1)).encode('utf-8')
            key_bytes = key_bytes[:len(encrypted_bytes)]

            # XOR Decrypt
            decrypted_bytes = bytes([encrypted_bytes[i] ^ key_bytes[i] for i in range(len(encrypted_bytes))])

            # Check Length
            if len(decrypted_bytes) != text_length:
                raise ValueError("Text Length Does Not Match!")

            return decrypted_bytes.decode('utf-8')
        except:
            raise ValueError("Decrypt Failed! Data Might be Destroyed or Incorrect Key!")

    def encrypt_dataframe(self, data):
        """
        Encrypt an entire pandas DataFrame.

        Convert all column values of the DataFrame to strings and encrypt them, and append the configured suffix to the column names.
        The method returns a new DataFrame; the original data is not modified.

        Parameters:
            data (pandas.DataFrame): pandas DataFrame to encrypt.
                                   The values of all columns are converted to strings before encryption.

        Returns:
            pandas.DataFrame: New encrypted DataFrame with the following properties:
                             - All column values are encrypted and Base64-encoded
                             - All column names carry the suffix specified at initialization (default '_encrypted')
                             - A copy is returned; the original DataFrame is unchanged

        Raises:
            AttributeError: If encryption fails because the key attribute is None.

        Note:
            - An encrypted DataFrame cannot be used directly for data analysis; it must be decrypted first
            - Back up the mapping of the original DataFrame column names before encrypting

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> encryptor = TextEncryptor(key="secret")
            >>> encrypted_df = encryptor.encrypt_dataframe(df)
            >>> print(encrypted_df.columns.tolist())  # Output: ['name_encrypted', 'age_encrypted']
        """
        res = data.copy()
        collist = data.columns.tolist()
        for col in collist:
            ## Encryption
            res[col] = res[col].astype(str)
            res[col] = res[col].apply(lambda x: self.encrypt(x))
        res.columns = [x + self.suffix for x in res.columns]
        return res

    def decrypt_dataframe(self, data):
        """
        Decrypt an encrypted pandas DataFrame.

        Iterate over all columns of the DataFrame, decrypt every column value, and remove the encryption suffix from the column names.
        The method returns a new DataFrame; the original data is not modified.

        Parameters:
            data (pandas.DataFrame): pandas DataFrame to decrypt.
                                   It should be a DataFrame produced by the encrypt_dataframe method.

        Returns:
            pandas.DataFrame: New decrypted DataFrame with the following properties:
                             - All column values are decrypted and restored to their original string form
                             - The suffix specified at initialization (default '_encrypted') is removed from all column names
                             - A copy is returned; the original DataFrame is unchanged

        Raises:
            ValueError: If decryption fails. Possible causes include:
                       - A column value is not a valid encrypted string
                       - The wrong key was used for decryption
                       - The data was corrupted during transmission or storage
            UnicodeDecodeError: If the decrypted bytes cannot be decoded as a UTF-8 string.

        Note:
            - Encryption and decryption must use the same key
            - Decryption may fail if the DataFrame contains columns that are not encrypted

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name_encrypted': ['aGVsbG8=', 'd29ybGQ='],
            ...                    'age_encrypted': ['c2F2ZWQ=', 'dGVzdA==']})
            >>> encryptor = TextEncryptor(key="secret")
            >>> decrypted_df = encryptor.decrypt_dataframe(df)
            >>> print(decrypted_df.columns.tolist())  # Output: ['name', 'age']
        """
        res = data.copy()
        collist = data.columns.tolist()
        for col in collist:
            ## Encryption
            res[col] = res[col].apply(lambda x: self.decrypt(x))
        res.columns = [x.replace(self.suffix, "") for x in res.columns]
        return res

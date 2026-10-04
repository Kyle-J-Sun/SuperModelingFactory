import base64
import random
import pandas as pd
import numpy as np

class TextEncryptor:
    """
    Text encryption and decryption utility class based on the XOR algorithm.

    The class encrypts and decrypts single strings and entire pandas DataFrames. Encrypted data is encoded as URL-safe
    Base64, which makes it easy to store and transmit. This is obfuscation, not strong cryptography: do not rely on it to
    protect secrets.

    Parameters
    ----------
    key : str, default None
        Key used for encryption and decryption. A key is required in practice: with ``None`` (the default) ``encrypt``
        raises ``TypeError``, with an empty string it raises ``ZeroDivisionError``, and ``decrypt`` raises ``ValueError``
        in both cases.
    suffix : str, default '_encrypted'
        Suffix that ``encrypt_dataframe`` appends to the column names. ``decrypt_dataframe`` removes every occurrence of
        it from the column names.

    Attributes
    ----------
    key : str
        The key given at construction.
    suffix : str
        The column-name suffix given at construction.

    Examples
    --------
    >>> encryptor = TextEncryptor(key="my_secret_key")
    >>> encrypted = encryptor.encrypt("Hello World")
    >>> encryptor.decrypt(encrypted)
    'Hello World'
    """

    def __init__(self, key=None, suffix='_encrypted'):
        """Store the key and the column-name suffix (see the class docstring for the parameters)."""
        self.key = key
        self.suffix = suffix

    def encrypt(self, text):
        """
        Encrypt a string.

        The UTF-8 bytes of ``text`` are XORed with the repeated key and prefixed with 2 bytes that hold the length of
        the plaintext in bytes, which ``decrypt`` uses for validation. The result is encoded as URL-safe Base64.

        Parameters
        ----------
        text : str
            Plaintext to encrypt, at most 65,535 bytes of UTF-8.

        Returns
        -------
        str
            Encrypted text, URL-safe Base64.

        Raises
        ------
        TypeError
            If the key is None.
        ZeroDivisionError
            If the key is an empty string.
        OverflowError
            If ``text`` is longer than 65,535 bytes.

        Examples
        --------
        >>> encryptor = TextEncryptor(key="secret")
        >>> encryptor.encrypt("Hello")
        'AAU7AA8eCg=='
        """
        # Text to bytes
        text_bytes = text.encode('utf-8')

        # Expand byte length
        key_bytes = (self.key * (len(text_bytes) // len(self.key) + 1)).encode('utf-8')
        key_bytes = key_bytes[:len(text_bytes)]

        # XOR encryption
        encrypted_bytes = bytes([text_bytes[i] ^ key_bytes[i] for i in range(len(text_bytes))])

        # Add 2 more bytes for verification: the plaintext length in bytes, which is what decrypt compares against
        length_byte = len(text_bytes).to_bytes(2, 'big')

        # combine
        result_bytes = length_byte + encrypted_bytes
        return base64.urlsafe_b64encode(result_bytes).decode('utf-8')

    def decrypt(self, encrypted_text):
        """
        Decrypt a string produced by ``encrypt``.

        The Base64 text is decoded, the first 2 bytes give the stored plaintext length, and the remaining bytes are
        XORed with the repeated key. The length of the recovered bytes is checked against the stored length.

        Parameters
        ----------
        encrypted_text : str
            URL-safe Base64 text produced by ``encrypt``.

        Returns
        -------
        str
            The original plaintext.

        Raises
        ------
        ValueError
            If decryption fails for any reason: the input is not valid Base64, the key is wrong or missing, the stored
            length does not match the recovered length (the data was altered), or the recovered bytes are not valid
            UTF-8.

        Examples
        --------
        >>> encryptor = TextEncryptor(key="secret")
        >>> encryptor.decrypt(encryptor.encrypt("Hello"))
        'Hello'
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
        Encrypt every value of a DataFrame.

        All values are converted to strings (a missing value becomes ``'nan'``) and encrypted with ``encrypt``, and the
        suffix is appended to every column name. The original DataFrame is not modified.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame to encrypt.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame of encrypted strings (object dtype) whose column names carry the suffix.

        Raises
        ------
        TypeError
            If the key is None.
        ZeroDivisionError
            If the key is an empty string.
        OverflowError
            If a value, converted to a string, is longer than 65,535 bytes.

        Notes
        -----
        An encrypted DataFrame cannot be analyzed; decrypt it first. A column name that already contains the suffix text
        does not survive a round trip, because ``decrypt_dataframe`` removes every occurrence of the suffix.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> encryptor = TextEncryptor(key="secret")
        >>> encryptor.encrypt_dataframe(df).columns.tolist()
        ['name_encrypted', 'age_encrypted']
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
        Decrypt a DataFrame produced by ``encrypt_dataframe``.

        Every value is decrypted with ``decrypt`` and every occurrence of the suffix is removed from the column names.
        The original DataFrame is not modified.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame produced by ``encrypt_dataframe`` with the same key.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame whose values are the decrypted strings (object dtype: numbers and missing values come back
            as text such as ``'2'`` and ``'nan'``) and whose column names no longer contain the suffix.

        Raises
        ------
        ValueError
            If a value cannot be decrypted, for example because the key is wrong or a column was never encrypted.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> encryptor = TextEncryptor(key="secret")
        >>> decrypted = encryptor.decrypt_dataframe(encryptor.encrypt_dataframe(df))
        >>> decrypted.columns.tolist()
        ['name', 'age']
        >>> decrypted['age'].tolist()
        ['25', '30']
        """
        res = data.copy()
        collist = data.columns.tolist()
        for col in collist:
            ## Encryption
            res[col] = res[col].apply(lambda x: self.decrypt(x))
        res.columns = [x.replace(self.suffix, "") for x in res.columns]
        return res

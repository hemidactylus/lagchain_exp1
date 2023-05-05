## Notes with SQL

Secondary details trying to use SQL databases for the LLM caching.

##### A MySQL failure

Let's try it with MySQL: first,

```
docker run --name local-mysql -e MYSQL_ROOT_PASSWORD=123456 -d mysql
```

then [more steps](https://www.howtogeek.com/devops/how-to-run-mysql-in-a-docker-container/) are to be done.

Log to the MySQL with `docker exec -it local-mysql mysql -p` and then in the mysql shell run:

```
create database CACHEDB;
create user 'cacher'@'%' identified by 'cachepwd';
grant all privileges on CACHEDB.* TO 'cacher'@'%';
flush privileges;
```

For Python, some massaging to get MySQL working

```
sudo apt-get install libmysqlclient-dev
sudo apt-get install python-dev default-libmysqlclient-dev
sudo apt install libffi-dev
sudo apt install python3.10-dev
pip install mysqlclient
```

or alternatively just

```
pip install pymysql
```

then change the connection string in the Python cell to either of:

```
# engine = create_engine("mysql+mysqldb://cacher:cachepwd@172.17.0.2/CACHEDB")
engine = create_engine("mysql+pymysql://cacher:cachepwd@172.17.0.2/CACHEDB")
```

But **FAILURE**, it seems the SQLAlchemy fails in required varchar length in this sql dialect:

```
(in table 'full_llm_cache', column 'prompt'): VARCHAR requires a length on dialect mysql
```

Too much time wasted already. Erase the container and try with another SQL DB.

##### Another SQL DB failure

MariaDB, let's see.

```
docker run -p 127.0.0.1:3306:3306  --name mdb -e MARIADB_ROOT_PASSWORD=Password123! -d mariadb:latest
docker exec -it mdb mariadb --user root -pPassword123!
# ... plus the same user/db creation as for mysql.
```

Connection string, see [here](https://github.com/zilliztech/GPTCache/blob/6a1e2e82aabcd3a48486042ef5c7c6323f8589fd/gptcache/manager/scalar_data/sql_storage.py#L99) for a survey:

```
engine = create_engine("mariadb+pymysql://cacher:cachepwd@localhost/CACHEDB")
```

Sadly, **same problem**.
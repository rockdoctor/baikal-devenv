FROM rockdoctor/hadoophive:latest

COPY config/hive-site.xml $HIVE_HOME/conf/

EXPOSE 9083

ENTRYPOINT [ "sh", "-c", "$HIVE_HOME/bin/schematool -initSchema -dbType postgres; $HIVE_HOME/bin/hive --service metastore; service ssh start; bash"]

FROM rockdoctor/hadoophive:latest

COPY config/hive-site.xml $HIVE_HOME/conf/

EXPOSE 10000 10002

ENTRYPOINT [ "sh", "-c", "$HIVE_HOME/bin/hive --service hiveserver2; service ssh start; bash"]

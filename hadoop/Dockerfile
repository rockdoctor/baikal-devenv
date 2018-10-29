FROM rockdoctor/hadoophive:latest

# For Cleanup at End
WORKDIR /tmp

# Install python
RUN apt-get -yqq install python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install useful python modules
RUN pip install numpy scipy cython pandas
RUN pip install https://github.com/scikit-learn/scikit-learn/zipball/master
RUN pip install hdfs confluent-kafka

# Add hadoop user
RUN groupadd hadoop
RUN useradd -d /home/hadoop -g hadoop -m hadoop

# Authorize SSH key for hadoop
RUN mkdir /home/hadoop/.ssh
RUN ssh-keygen -t rsa -f /home/hadoop/.ssh/id_rsa -P '' && \
    cat /home/hadoop/.ssh/id_rsa.pub >> /home/hadoop/.ssh/authorized_keys

# Install Scala
RUN wget -q http://downloads.lightbend.com/scala/2.11.11/scala-2.11.11.tgz
RUN tar -xzf scala-2.11.11.tgz -C /usr/local/
RUN mv /usr/local/scala-2.11.11 /usr/local/scala
RUN chown -R root:root /usr/local/scala
ENV SCALA_HOME=/usr/local/scala

# Install Spark and link python with pyspark
RUN wget -q https://archive.apache.org/dist/spark/spark-2.3.2/spark-2.3.2-bin-without-hadoop.tgz
RUN tar -xzf spark-2.3.2-bin-without-hadoop.tgz -C /usr/local/
RUN mv /usr/local/spark-2.3.2-bin-without-hadoop /usr/local/spark
RUN wget -q http://central.maven.org/maven2/org/apache/spark/spark-hive_2.11/2.3.2/spark-hive_2.11-2.3.2.jar
RUN mv spark-hive_2.11-2.3.2.jar /usr/local/spark/jars/spark-hive_2.11-2.3.2.jar
RUN wget -q http://central.maven.org/maven2/org/apache/spark/spark-hive-thriftserver_2.11/2.3.2/spark-hive-thriftserver_2.11-2.3.2.jar
RUN mv spark-hive-thriftserver_2.11-2.3.2.jar /usr/local/spark/jars/spark-hive-thriftserver_2.11-2.3.2.jar
ENV SPARK_HOME=/usr/local/spark
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH
RUN pip install pyspark
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
ENV PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

# Configure Hadoop classpath for Spark
RUN echo "export SPARK_DIST_CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath):/usr/local/hive/lib/*" > /usr/local/spark/conf/spark-env.sh

# Install Zeppelin
RUN wget -q http://archive.apache.org/dist/zeppelin/zeppelin-0.7.3/zeppelin-0.7.3-bin-netinst.tgz
RUN tar -xzf zeppelin-0.7.3-bin-netinst.tgz -C /usr/local/
RUN mv /usr/local/zeppelin-0.7.3-bin-netinst /usr/local/zeppelin
ENV ZEPPELIN_HOME=/usr/local/zeppelin
COPY config/zeppelin-env.sh $ZEPPELIN_HOME/conf/zeppelin-env.sh
RUN chown -R hadoop:hadoop $ZEPPELIN_HOME

# Set the PATH environment variable globally and for the Hadoop user
ENV PATH=$PATH:$SCALA_HOME/bin:$SPARK_HOME/bin:$ZEPPELIN_HOME/bin
RUN echo "PATH=$PATH:$SCALA_HOME/bin:$SPARK_HOME/bin:$ZEPPELIN_HOME/bin" >> /home/hadoop/.bashrc

# Hadoop configuration
COPY config/sshd_config /etc/ssh/sshd_config
COPY config/ssh_config /home/hadoop/.ssh/config
COPY config/hadoop-env.sh config/hdfs-site.xml config/hdfs-site.xml config/core-site.xml \
     config/core-site.xml config/mapred-site.xml config/yarn-site.xml config/yarn-site.xml \
     $HADOOP_CONF_DIR/

# Initialization scripts
RUN mkdir $HADOOP_HOME/bin/init
COPY init-scripts/init-hadoop.sh $HADOOP_HOME/bin/init/
COPY init-scripts/start-hadoop.sh init-scripts/stop-hadoop.sh $HADOOP_HOME/bin/init/
COPY init-scripts/hadoop /etc/init.d/

# Utilities
RUN mkdir -p /home/hadoop/utils
COPY utils/run-wordcount.sh utils/format-namenode.sh /home/hadoop/utils/

# Replace Hadoop slave file with provided one and changing logs directory
RUN rm $HADOOP_CONF_DIR/slaves
RUN ln -s /config/slaves $HADOOP_CONF_DIR/slaves

# Set up log directories
RUN ln -s /data/logs/hadoop $HADOOP_HOME/logs
RUN ln -s $HADOOP_HOME/logs /var/log/hadoop
RUN ln -s $ZEPPELIN_HOME/logs /var/log/zeppelin

# Set permissions on Hadoop home
RUN chown -R hadoop:hadoop $HADOOP_HOME
RUN chown -R hadoop:hadoop /home/hadoop
RUN chmod 644 /home/hadoop/.ssh/config

# Configure Hive
COPY config/hive-site.xml $HIVE_HOME/conf/
COPY config/hive-site.xml $SPARK_HOME/conf/hive-site.xml
COPY config/hive-site.xml $ZEPPELIN_HOME/conf/hive-site.xml

# Configure Spark links for Hive
RUN ln -s /usr/local/spark/jars/spark-network-common_2.11-2.2.0.jar /usr/local/hive/lib/spark-network-common_2.11-2.2.0.jar
RUN ln -s /usr/local/spark/jars/spark-core_2.11-2.2.0.jar /usr/local/hive/lib/spark-core_2.11-2.2.0.jar
RUN ln -s /usr/local/spark/jars/scala-library-2.11.8.jar /usr/local/hive/lib/scala-library-2.11.8.jar

RUN chown -R hadoop:hadoop $SPARK_HOME/conf/

# Cleanup
RUN rm -rf /tmp/*
RUN apt-get clean

WORKDIR /root

EXPOSE  2222 4040 8020 8030 8031 8032 8033 8042 8088 9001 50010 50020 50070 50075 50090 50100

VOLUME /config
VOLUME /deployments

ENTRYPOINT [ "sh", "-c", "service ntp start; $HADOOP_HOME/bin/init/init-hadoop.sh; service ssh start; bash"]

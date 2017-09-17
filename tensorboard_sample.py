import tensorflow as tf

cross_entropy = tf.placeholder(tf.float32)

# sessionの用意
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
#sess.run(tf.global_variables_initializer())

# summaryの設定
tf.summary.scalar('cross_entropy', cross_entropy)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('shiba_train', sess.graph)

# 100回実行してcross_entropyのsummaryを記録
for step in range(100):
    summary_str = sess.run(summaries, {cross_entropy: step})
    train_writer.add_summary(summary_str, step)

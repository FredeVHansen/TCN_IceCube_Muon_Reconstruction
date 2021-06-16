{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import sklearn as sk\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import roc_curve\n",
    "from sklearn.metrics import auc\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tcn import TCN\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import sqlite3\n",
    "import time\n",
    "import pickle as pkl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#path = './Event_files/'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('azimuth_500k_train_all.pkl','rb') as f:\n",
    "    train_events = pkl.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('azimuth_200k_val_all.pkl','rb') as f:\n",
    "    validation_events = pkl.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('200k_zenith_val_final_event_no.pkl','rb') as f:\n",
    "    event_no = pkl.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "500 200\n"
     ]
    }
   ],
   "source": [
    "vallen = len(validation_events)\n",
    "print(len(train_events), len(validation_events))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "##needed for validation\n",
    "#X_valQ, y_valQ = [], []\n",
    "#for i in range(vallen):\n",
    "#    X_valQ.append(validation_events[i][0])\n",
    "#    y_valQ.append(validation_events[i][1])\n",
    "#    \n",
    "#for i in range(vallen):\n",
    "#    lens = []\n",
    "#    lens.append(X_valQ[i][:,0,0].shape)\n",
    "##    print(min(lens[0]))\n",
    "#    min_len = int(min(lens[0]))\n",
    "#    \n",
    "#X_val, y_val = [], []\n",
    "#for i in range(vallen):\n",
    "#    X_val.append(np.array(X_valQ[i][:min_len, :, :]))\n",
    "#    y_val.append(np.array(y_valQ[i][:min_len,:]))\n",
    "#    \n",
    "##y_val = np.concatenate(y_val)\n",
    "##X_val = np.concatenate(X_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#QQ = []\n",
    "#for i in range(20):\n",
    "#    QQ.append(np.arccos(y_val[i][:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plt.hist(np.concatenate(QQ));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def TransformTruthToInput(truth):\n",
    "#    truth = truth.values[0][0]\n",
    "    #out = tf.convert_to_tensor([tf.math.sin(truth) , tf.math.cos(truth)])\n",
    "#    truth = 10**truth\n",
    "    out = [np.sin(truth), np.cos(truth)]\n",
    "    return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Skræddersyet funktion af model.fit\n",
    "#def Train(model, batches):\n",
    "#    Training_Losses = []\n",
    "#    Validation_Losses = [10e5]\n",
    "#    patience = 0\n",
    "#\n",
    "#    for epoch in range(n_epochs):       \n",
    "#        if patience < 5:\n",
    "#            epoch_loss = 0\n",
    "#            print('TRAINING EPOCH: %s / %s'%(epoch+1, n_epochs))\n",
    "#            for batch in batches:\n",
    "#                batch_loss = model.train_on_batch(x = batch[0], y = batch[1])\n",
    "#                Training_Losses.append(batch_loss)\n",
    "#                epoch_loss +=batch_loss\n",
    "#                    \n",
    "#            val_loss = model.evaluate(x = X_val, y = y_val, verbose=0)\n",
    "#            \n",
    "#            if val_loss < min(Validation_Losses):\n",
    "#                model.save('Earlystop'+str(vallen))\n",
    "#                patience = 0\n",
    "#            else:\n",
    "#                patience += 1\n",
    "#    \n",
    "#            Validation_Losses.append(val_loss)\n",
    "#            print('Epoch Loss:', np.round(epoch_loss,2))\n",
    "#            print('Last 5 Validation Losses:', np.round(Validation_Losses[1:][-5:],4))\n",
    "#    \n",
    "#    print('training done!')\n",
    "#    return model, Training_Losses, Validation_Losses\n",
    "\n",
    "#Skræddersyet funktion af model.fit\n",
    "def Train(model, batches):# events, batch_size, db_file, max_length, n_epochs, max_event_size):\n",
    "    #n_batches = int(len(events)/batch_size)\n",
    "    #print('Getting %s batches'%(n_batches))\n",
    "    #event_list = np.array_split(events,n_batches)\n",
    "    #k = 1\n",
    "    #batches = []\n",
    "    #for event_batch in event_list:\n",
    "    #    print('Getting Batch %s / %s'%(k,n_batches))\n",
    "    #    batch_features, batch_truth, _ = FixMyInput(event_batch['event_no'].reset_index(drop = True), db_file, max_length, max_event_size)\n",
    "    #    batches.append([batch_features, batch_truth])\n",
    "    #    k +=1\n",
    "    Training_Losses = []\n",
    "    #Training_Losses\n",
    "    for epoch in range(n_epochs):\n",
    "        epoch_loss = 0\n",
    "        print('TRAINING EPOCH: %s / %s'%(epoch+101, n_epochs))\n",
    "        \n",
    "        for batch in batches:\n",
    "            #print(batch[0].shape)\n",
    "            #print(np.shape(batch[1]))\n",
    "           \n",
    "            batch_loss = model.train_on_batch(x = batch[0], y = batch[1])\n",
    "            #model.train_on_batch(x = batch[0], y = batch[1])\n",
    "            Training_Losses.append(batch_loss)\n",
    "            epoch_loss +=batch_loss\n",
    "            \n",
    "        print(epoch_loss)\n",
    "        \n",
    "        model.save('Model_at_epoch'+str(epoch+100))\n",
    "        with open('Loss_at_'+str(epoch+100)+'.pkl','wb') as f:\n",
    "            pkl.dump(Training_Losses, f)\n",
    "            \n",
    "    print('training done!')\n",
    "    return model, Training_Losses\n",
    "\n",
    "\n",
    "\n",
    "#def TransformTruthToInput(truth):\n",
    "  #  out = [np.sin(truth), np.cos(truth)]\n",
    "  #  return out\n",
    "\n",
    "def sincosloss(y_true, y_pred): # y_true = [sin(zenith), cos(zenith)]\n",
    "    thetruth = tf.math.atan2(y_true[:,0], y_true[:,1])\n",
    "    thepred = tf.math.atan2(y_pred[:,0], y_pred[:,1])\n",
    "    co = 1- tf.math.cos(thetruth-thepred)\n",
    "    co = tf.reduce_sum(co)\n",
    "    \n",
    "    return co"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "##### MODEL CONFIGURATION\n",
    "n_filters = 32\n",
    "kernel_size = 3\n",
    "output_dim = 2\n",
    "lr = 1e-3\n",
    "batch_size = 1000\n",
    "n_epochs = 100\n",
    "max_event_size = 250\n",
    "max_length = max_event_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 250, 7)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_events[0][0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "acti = tf.keras.layers.LeakyReLU(alpha=0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tcn_model_26(max_length, N_filters, kernel_size, output_dim):\n",
    "    i = tf.keras.Input(batch_shape = (None, max_length, 7))\n",
    "    o = TCN(nb_filters = n_filters, #dilations = [1,2,4,8,16,32], \n",
    "            kernel_size = kernel_size, dropout_rate = 0.001, activation = acti)(i)\n",
    "    o = tf.keras.layers.Dense(n_filters, activation = acti)(o)\n",
    "    o = tf.keras.layers.Dense(n_filters, activation = acti)(o) \n",
    "    o = tf.keras.layers.Dropout(0.001)(o)\n",
    "    o = tf.keras.layers.Dense(n_filters, activation = acti)(o) \n",
    "    #o  = TCN(nb_filters = 32, kernel_size = kernel_size, dropout_rate=0.001, activation = 'relu')(o)\n",
    "    \n",
    "    o = tf.keras.layers.Dense(output_dim, activation='tanh')(o)\n",
    "    model = tf.keras.models.Model(inputs=[i], outputs=[o])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tcn_model_26(max_length, n_filters, kernel_size, output_dim)\n",
    "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss = sincosloss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#import tensorflow.python.util.deprecation as deprecation\n",
    "#deprecation._PRINT_DEPRECATION_WARNINGS = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "model-loaded\n"
     ]
    }
   ],
   "source": [
    "model.load_weights('Model_at_epoch89/variables/variables')\n",
    "trained_model = model\n",
    "print('model-loaded')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    " Træningsfunktion\n",
    "\n",
    "with tf.device('/GPU:0'):\n",
    "    start = time.time()\n",
    "    trained_model, training_losses = Train(model, train_events)\n",
    "    end = time.time()\n",
    "    print(round(end-start,3),'sec')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#fig, ax = plt.subplots(2,1, figsize=(8,10))\n",
    "#ax[0].plot(training_losses[100:])\n",
    "#ax[0].set_title('Training Loss')\n",
    "#ax[0].set_ylabel('Loss')\n",
    "#ax[0].set_xlabel('Epoch')\n",
    "#ax[1].set_ylabel('Loss')\n",
    "#ax[1].set_xlabel('Epoch')\n",
    "#ax[1].plot(validation_losses[1:])\n",
    "#ax[1].set_title('Validation Loss');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#trained_model = tf.keras.models.load_model('Az_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#def Predict(model, X, y):#, batch_size): Predict function looks like this for 2D output\n",
    "#    predictions = []\n",
    "#    truth = []\n",
    "#    for i in range(len(y)):\n",
    "#        out = model.predict(X[i])\n",
    "#        predictions.append(out)\n",
    "#        truth.append(y[i])\n",
    "#    print('prediction done!')\n",
    "#    length = 1000 * len(truth)\n",
    "#    predictions = np.reshape(np.array(predictions), (length,2))\n",
    "#    truth = np.reshape(np.array(truth), (length,2))\n",
    "#    return predictions, truth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Predict(model, batches):#, batch_size):\n",
    "    predictions = []\n",
    "    truth = []\n",
    "    for event_batch in batches:\n",
    "        out = model.predict(event_batch[0])\n",
    "        predictions.extend(out)\n",
    "        truth.extend(event_batch[1])\n",
    "    print('prediction done!')\n",
    "    return predictions, truth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "#with tf.device('/GPU:0'):\n",
    "#    start = time.time()\n",
    "#    \n",
    "#    pred,truth  = Predict(trained_model, X_val, y_val)#, batch_size)\n",
    "#    \n",
    "#    end = time.time()\n",
    "#    \n",
    "#    print(round(end-start,3),'sec')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "prediction done!\n",
      "74.647 sec\n"
     ]
    }
   ],
   "source": [
    "with tf.device('/GPU:0'):\n",
    "    start = time.time()\n",
    "    \n",
    "    pred,truth  = Predict(trained_model, validation_events)#, batch_size)\n",
    "    \n",
    "    end = time.time()\n",
    "    \n",
    "    print(round(end-start,3),'sec')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.device('/GPU:0'):\n",
    "    azimuth_pred = []\n",
    "    azimuth      = []\n",
    "    for i in range(0,len(pred)):\n",
    "        azimuth_pred.append(np.arctan2(pred[i][0],pred[i][1]))\n",
    "        azimuth.append(np.arctan2(truth[i][0],truth[i][1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "correlation: \n",
      "[[1.         0.65400824]\n",
      " [0.65400824 1.        ]]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAD4CAYAAAAAczaOAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAgh0lEQVR4nO3df7AV5Z3n8fdHQoSJij9AglzYy2YwqCRD9BYhA7XrqInEZEaT0g1OjKTGDYZoTZLN7gZnq5I7idQyG6Ib3dEUSSxw1h+LFbOiwST+iJXoiHh1UQQk4krMFSKGCQbLyCJ+94/TB5vLOff2ubfPz/68qm6dPk93n/Occ/p+++lvP92PIgIzMyuGI5pdATMzaxwHfTOzAnHQNzMrEAd9M7MCcdA3MyuQdzS7AkMZP358dHd3N7saZmZt5YknnvhdREwYWN7yQb+7u5u+vr5mV8PMrK1I+nWlcqd3zMwKxEHfzKxAHPTNzAqk5XP6Zma12L9/P/39/bzxxhvNrkpDjBkzhq6uLkaPHp1peQd9M+so/f39HH300XR3dyOp2dWpq4hg9+7d9Pf3M23atEzrOL1jZh3ljTfe4IQTTuj4gA8giRNOOKGmoxoHfTPrOEUI+GW1flYHfTOzAnFO38w62txlD/LSnj/m9nqTjx3LI0vOqjp/z5493HrrrXzhC1+o6XVXrlzJRz7yEU466STg7QtTx48fP6L6DuSgb4e79n3w6oul6XFT4csbm1sfsxF4ac8f2b7sY7m9XveSHw86f8+ePdxwww2HBf0DBw4watSoquutXLmSmTNnHgz69eKgb4d79UXofbU03TuuuXUxazNLlizh+eefZ9asWYwePZqjjjqKSZMmsWHDBtauXcvHP/5xnnnmGQCWL1/Oa6+9xsyZM+nr6+PTn/40Y8eO5dFHHwXg+uuv5+6772b//v3ccccdzJgxY8T1c07fzCxHy5Yt4z3veQ8bNmzgW9/6FuvXr2fp0qVs3ry56joXXnghPT093HLLLWzYsIGxY8cCMH78eJ588kkWL17M8uXLc6mfg76ZWR3Nnj07cx/6gT75yU8CcMYZZ7B9+/Zc6uOgb2ZWR+9617sOTr/jHe/grbfeOvh8qP71Rx55JACjRo3izTffzKU+mYO+pFGS/o+ke5Lnx0u6T9JzyeNxqWWvkrRN0lZJ56bKz5C0MZl3nYrUmbbVXfu+Uv6+d1zp5K2ZDcvRRx/N3r17K86bOHEiu3btYvfu3ezbt4977rkn03p5quVE7heBLcAxyfMlwAMRsUzSkuT5VyWdCiwATgNOAu6XdHJEHABuBBYB64C1wHzg3lw+iY1M+uStWQeZfOzYIXvc1Pp6gznhhBOYO3cuM2fOZOzYsUycOPHgvNGjR/O1r32ND37wg0ybNu2QE7Of/exn+fznP3/Iidx6UEQMvZDUBawClgL/ISI+LmkrcGZE7JQ0CXgoIt4r6SqAiPivybo/BXqB7cDPI2JGUn5xsv7lg713T09PeBCVBugdVznoVys3a1FbtmzhlFNOaXY1GqrSZ5b0RET0DFw2a3rnvwP/GXgrVTYxInYCJI8nJuWTgd+klutPyiYn0wPLDyNpkaQ+SX2vvPJKxiqamdlQhgz6kj4O7IqIJzK+ZqU8fQxSfnhhxIqI6ImIngkTDhvi0Rpp3NS3c/3Xvq/ZtTGzEcqS058L/JWk84AxwDGS/ifwsqRJqfTOrmT5fmBKav0uYEdS3lWh3FpZ+mpcX6hl1vaGbOlHxFUR0RUR3ZRO0D4YEZcAa4CFyWILgbuS6TXAAklHSpoGTAfWJymgvZLmJL12Lk2tY2ZmDTCS2zAsA1ZLugx4EbgIICI2SVoNbAbeBK5Ieu4ALAZWAmMp9dpxzx0zswaqKehHxEPAQ8n0buDsKsstpdTTZ2B5HzCz1kpanQy8sdpQyvn98rRvxGbWdnzDtSKrtW++8/vWjtKNmzw0uMHz0EMPsXz58kMu5BoJB30z62x5X3iYU4NnqFst14vvvWNmlrPt27czY8YMFi5cyPvf/34uvPBCXn/9dbq7u/nGN77BvHnzuOOOO/jZz37Ghz70IU4//XQuuugiXnvtNQB+8pOfMGPGDObNm8edd96Za90c9M3M6mDr1q0sWrSIp59+mmOOOYYbbrgBgDFjxvDwww9zzjnncPXVV3P//ffz5JNP0tPTwzXXXMMbb7zB5z73Oe6++25++ctf8tvf/jbXejnom5nVwZQpU5g7dy4Al1xyCQ8//DAAn/rUpwBYt24dmzdvZu7cucyaNYtVq1bx61//mmeffZZp06Yxffp0JHHJJZfkWi/n9M3M6mDgTYTLz8u3Wo4IPvzhD3PbbbcdstyGDRsOWzdPbumbmdXBiy++ePBumbfddhvz5s07ZP6cOXN45JFH2LZtGwCvv/46v/rVr5gxYwYvvPACzz///MF18+SWvpl1tvT1JXm9XgannHIKq1at4vLLL2f69OksXryY66+//uD8CRMmsHLlSi6++GL27dsHwNVXX83JJ5/MihUr+NjHPsb48eOZN2/ewTF18+Cgb2adrUkXER5xxBF897vfPaRs4JCHZ511Fo8//vhh686fP59nn322PvWqy6uamVlLctA3M8tZd3d3rimZPDnom1nHyTIiYKeo9bM66JtZRxkzZgy7d+8uROCPCHbv3s2YMWMyr+MTuWbWUbq6uujv76coQ62OGTOGrq6uoRdMOOibWUcZPXo006ZNa3Y1WpbTO2ZmBZJlYPQxktZLekrSJkl/n5T3SnpJ0obk77zUOldJ2iZpq6RzU+VnSNqYzLtO9bzW2MzMDpMlvbMPOCsiXpM0GnhYUnmYw2sjYnl6YUmnUhpL9zTgJOB+SScnQybeCCwC1gFrgfl4yEQzs4bJMjB6RMRrydPRyd9gp8XPB26PiH0R8QKwDZgtaRJwTEQ8GqXT6jcDF4yo9mZmVpNMOX1JoyRtAHYB90XEY8msKyU9LekmScclZZOB36RW70/KJifTA8srvd8iSX2S+opyBt7MrBEyBf2IOBARs4AuSq32mZRSNe8BZgE7gW8ni1fK08cg5ZXeb0VE9EREz4QJE7JU0czMMqipy2ZE7JH0EDA/ncuX9D2gPGpvPzAltVoXsCMp76pQbmZ26ADmDR58vEiy9N6ZIOnYZHoscA7wbJKjL/sEUL7RxBpggaQjJU0DpgPrI2InsFfSnKTXzqXAXfl9FDNra+UBzHtffTv4W+6ytPQnAaskjaK0k1gdEfdI+idJsyilaLYDlwNExCZJq4HNwJvAFUnPHYDFwEpgLKVeO+65Y1ZkA1v3VndDBv2IeBr4QIXyzwyyzlJgaYXyPmBmjXU0s05Vbt1bw/iKXDOzAvG9d8zMKunQE8sO+mZmlaRTT3mOsdtkDvo2POnBpjuoFWTW6Rz0bXjSQb6DWkFmnc5B38way900m8pB38way900m8pdNs3MCsRB38ysQBz0zcwKxDl9M6s/n7xtGQ76ZlZ/PnnbMpzeMTMrELf0zaz1+IrvunHQN7PW4yu+68bpHTOzAskyXOIYSeslPSVpk6S/T8qPl3SfpOeSx+NS61wlaZukrZLOTZWfIWljMu+6ZNhEa3flQ/HecaVeGmZ58vaVqyzpnX3AWRHxmqTRwMOS7gU+CTwQEcskLQGWAF+VdCqwADgNOAm4X9LJyZCJNwKLgHXAWmA+HjKx/flQ3OrJ21euhmzpR8lrydPRyV8A5wOrkvJVwAXJ9PnA7RGxLyJeALYBs5OB1I+JiEcjIoCbU+uYmVkDZDqRmwyK/gTwp8A/RsRjkiZGxE6AiNgp6cRk8cmUWvJl/UnZ/mR6YHml91tE6YiAqVN9IYe1sQ4dfcnaV6YTuRFxICJmAV2UWu2DDW5eKU8fg5RXer8VEdETET0TJkzIUkWz1lS+KKn31beDv1kT1dRlMyL2SHqIUi7+ZUmTklb+JGBXslg/MCW1WhewIynvqlBuZp2oHrdecP/9ERsy6EuaAOxPAv5Y4BzgH4A1wEJgWfJ4V7LKGuBWSddQOpE7HVgfEQck7ZU0B3gMuBS4Pu8PZGYtoh63XvBJ3RHL0tKfBKxK8vpHAKsj4h5JjwKrJV0GvAhcBBARmyStBjYDbwJXJD13ABYDK4GxlHrtuOeOmVkDDRn0I+Jp4AMVyncDZ1dZZymwtEJ5HzDY+QAzK/NJYKsD34bBrFWl0yNOZVhOHPTNLD++b37Lc9A3s/z4vvktzzdcMzMrEAd9M7MCcXrHbCh59aLJcmGRc+JWZw76ZkPJqxdNlguLnBO3OnPQt3ylW7MDy93P3KzpHPQtX9UC+7Xv8z1TOpVTUm3FQd8aw/dM6VxOSbUV994xMysQt/TNrHZO6bQtB32zZhjshHc7cEqnbTnoF41baK3BJ7JHzgOqDIuDftG0QgvN/6ztqdUaDO4cMCwO+tZ4/mdtT63QYLARG7L3jqQpkn4uaYukTZK+mJT3SnpJ0obk77zUOldJ2iZpq6RzU+VnSNqYzLtOUqXB0s1aV/kopXdcqeXbDu9bvkai0XW2lpSlpf8m8JWIeFLS0cATku5L5l0bEcvTC0s6FVgAnEZpjNz7JZ2cDJl4I7AIWAespTTAuodMtOpabfSoZh2ljOR90y309EVyaa3w3VpDZBkucSewM5neK2kLMHmQVc4Hbo+IfcALkrYBsyVtB46JiEcBJN0MXICDvg2mHUePalbuO8sOslpgb5fv1kasppy+pG5K4+U+BswFrpR0KdBH6Wjg95R2COtSq/UnZfuT6YHlld5nEaUjAqZObYETRtaemnWU0KzcdzvuIK3hMgd9SUcBPwS+FBF/kHQj8E0gksdvA38DVMrTxyDlhxdGrABWAPT09FRcxmrQar0uGqVTg6B7P9kIZAr6kkZTCvi3RMSdABHxcmr+94B7kqf9wJTU6l3AjqS8q0K51Vu79Lpotfx9q8qS3x+4YxiuojYYOtiQQT/pYfMDYEtEXJMqn5Tk+wE+ATyTTK8BbpV0DaUTudOB9RFxQNJeSXMopYcuBa7P76NYWxoYnOrVMi9a6zivz9cuDQbLLEtLfy7wGWCjpA1J2d8BF0uaRSlFsx24HCAiNklaDWym1PPniqTnDsBiYCUwltIJXJ/ELbpGBd9697ppVos4rxa9FUaW3jsPUzkfv3aQdZYCSyuU9wEza6mg2UGt1lqvdpTSSPX4DpzS6Wi+ItfaR6tdydvsnU69OKXT0Rz0rTWVW9FuaTaG00SF4aBvramerehWSxO1ghb/DuYue5CX9vwRgMnHjuWRJWc1uUbty0HfiqfV0kQFVGsQf2nPH9m+7GMAdC/5cd3r18kc9K39uX9/28kSxAfuGNpVqx2lOOhbe2pU/35rmvSOoZW0+1GKg761J7fmrUna/SjFQd/MmmrysWMrBs9qwbLS8tvH1KVqw9aqRyngoG/DVO0QN12e1rBcprsetp1at4uKy/fmU5da1brDKmtmnt9B3zIbuKFWOsSt1sJpWC6zStqn2s4oawuxXP9WOBFnh+uP8XQ1oRturdtCeSdR7f+nERz0LbNWPmQtq9aCqlr33myv20on4trJYEd+eZq377q3f98WPpnfCg0GB307TL0PPev5+sPpKeFWfP20Q0NhMOVto9XOGYyEg36La0bur95dzBrVhS2db63ashw3le38NQA735hA95LvHLZIJ/3DN0Izeq6kf+tDfq9ar+FILd8f41NHDwy9LbUJB/0myhLQW62P73BlCsA5y7SDTAWBSb3jRpQCarRWu+inrBmt+/Rn7//6gPx+LddwpG42N2/Jj9memlWvzzTwf6Pev6ODfhMVKaC3SkDqJJ2y/eTtkPx+G0j/bzTid8wyctYU4Gbg3cBbwIqI+I6k44H/BXRTGkTl3yUDoyPpKuAy4ADwtxHx06T8DN4eRGUt8MWI6PgxcGttkbXyhR2VFCmgN7pVllW1ejW9C20b6pQ0TjVZWvpvAl+JiCclHQ08Iek+4LPAAxGxTNISYAnwVUmnAguA0ygNl3i/pJOT0bNuBBYB6ygF/fkUYPSsdIts7rIHK25UA/9pG91SabcdTbM0ulWWVbV6Nb0LbRtqp6OE4cgyctZOYGcyvVfSFmAycD5wZrLYKuAh4KtJ+e0RsQ94QdI2YLak7cAxEfEogKSbgQsoQNBPq9a6ytLqqmcrs1pwaEYu3qrLa+fcqkcseejkz5aHmnL6krqBD1Aa2HxieWD0iNgp6cRkscmUWvJl/UnZ/mR6YHlHqkfLOb3xDjxiOGzDzmnIO//DVFfP4FLz9QaD1KuSVj1iyUM7f7ZG7LAyB31JRwE/BL4UEX+QKg2bW1q0QlkMUl7pvRZRSgMxdWp7Xkpf7x4MQ27YBRryrlmpqXoGl5GcpPWOun01YoeVKehLGk0p4N8SEXcmxS9LmpS08icBu5LyfmBKavUuYEdS3lWh/DARsQJYAdDT09PxJ3pHquiHs61wAVA7/waV7h9T62fwOaH2kaX3joAfAFsi4prUrDXAQmBZ8nhXqvxWSddQOpE7HVgfEQck7ZU0h1J66FLg+tw+SQvoxBZnvbVzsEyr529Q7/Mqlb7zWj9DK+x4LZssLf25wGeAjZI2JGV/RynYr5Z0GfAicBFARGyStBrYTKnnzxVJzx2AxbzdZfNeOuAkbrWbkFk27bzDylu1RkO77gitNWXpvfMwlfPxAGdXWWcpsLRCeR8ws5YKtrp2buG02iF5p7T6h6udtyVrH74iN9Gql7TXU6sFGbf6zerPQT/hS9rtEDl1eTVrNUc0uwKWr3SKZO6yB5tcmzZW7vLa+2rNA3KUfwN//9aK3NIfhlbLhacdTJH0UvGeK1Z/5d+g048YW/n/wKpz0B+GVsuFmzVDO/wfVL3Pfkp651WEsRMc9M3sMJ3Sk+qQevdWXuaQnVeVZTqJg75Z2bipbw+00aCTt62aInFPqs7loF9Bp7RyKmnVIFOrunyOGk/Y5qEdUiTWWRz0K+jkVk6nBJlO+RxmjeYum2ZmBeKWfg3KKYV2SYt0cprKLC8PH/m30PvXpScFuBDPQb8G7ZZS6OQ0VTvoxJ1uO58T6o/xdKVP1CfncLr0u8KMPQEO+kPycIE2XJ240223hk/avH3XHaz7zt4/ZVKyA9jJBCY1s2IN5qA/hE5oncHbOy/vuJqv3VrLndjw+dAb3zm4AyhSwAcH/cLolJ1XJ2i31nKnbDuduPMaDgd9MyuETtl5jdSQXTYl3SRpl6RnUmW9kl6StCH5Oy817ypJ2yRtlXRuqvwMSRuTeddpkJHVzXynSrP6yNLSXwn8D+DmAeXXRsTydIGkU4EFwGmUxse9X9LJyXCJNwKLgHXAWmA+HTBcotVHpTtVtlsuPM2pBWsVWYZL/IWk7oyvdz5we0TsA16QtA2YLWk7cExEPAog6WbgAhz089XhA3+0Wy48zakFaxUjyelfKelSoA/4SkT8HphMqSVf1p+U7U+mB5ZXJGkRpaMCpk7tvOBVN+WBP8zMqhjubRhuBN4DzAJ2At9Oyivl6WOQ8ooiYkVE9EREz4QJE4ZZRTMzG2hYLf2IeLk8Lel7wD3J035gSmrRLmBHUt5VodxsUM6Fm+VrWEFf0qSI2Jk8/QRQ7tmzBrhV0jWUTuROB9ZHxAFJeyXNAR4DLgWuH1nVrQicCzfL15BBX9JtwJnAeEn9wNeBMyXNopSi2Q5cDhARmyStBjYDbwJXJD13ABZT6gk0ltIJXJ/ENTNrsCy9dy6uUPyDQZZfCiytUN4HzKypdja0Du+xY2b58hW57c49dsysBh5ExcysQBz0zcwKxEHfzKxAHPTNzArEJ3LNzIYybipUGGqxHTnom5kNJR3ky8G/TTm9Y2ZWIA76ZmYF4qBvZlYgzum3soG3WGjjk0dm1hoc9FtZ+hYLbX7yyMxag9M7ZmYF4qBvZlYgTu+0Gt8q2czqyEG/1VS7VXIHXRFoZs0zZHpH0k2Sdkl6JlV2vKT7JD2XPB6XmneVpG2Stko6N1V+hqSNybzrJFUaLN2q+fLG0s4gfWK3d5yPBsysJlly+iuB+QPKlgAPRMR04IHkOZJOBRYApyXr3CBpVLLOjcAiSuPmTq/wmpZVegfgFr+Z1WDIoB8RvwD+ZUDx+cCqZHoVcEGq/PaI2BcRLwDbgNmSJgHHRMSjERHAzal1zMysQYbbe2diROwESB5PTMonA79JLdeflE1OpgeWVyRpkaQ+SX2vvPLKMKtoZmYD5d1ls1KePgYprygiVkRET0T0TJgwIbfKmZkV3XCD/stJyobkcVdS3g9MSS3XBexIyrsqlJuZWQMNN+ivARYm0wuBu1LlCyQdKWkapRO265MU0F5Jc5JeO5em1jEzswYZsp++pNuAM4HxkvqBrwPLgNWSLgNeBC4CiIhNklYDm4E3gSsi4kDyUosp9QQaC9yb/JmZWQMNGfQj4uIqs86usvxSYGmF8j5gZk21MzOzXPneO2ZmBeKgb2ZWIA76ZmYF4huutQLfWdPMGsRBvxVUu7OmmVnOnN4xMysQB30zswJx0DczKxAHfTOzAnHQNzMrEAd9M7MCcdA3MysQB30zswJx0DczKxAHfTOzAnHQNzMrkBEFfUnbJW2UtEFSX1J2vKT7JD2XPB6XWv4qSdskbZV07kgrb2Zmtcmjpf8XETErInqS50uAByJiOvBA8hxJpwILgNOA+cANkkbl8P5mZpZRPdI75wOrkulVwAWp8tsjYl9EvABsA2bX4f3NzKyKkQb9AH4m6QlJi5KyiRGxEyB5PDEpnwz8JrVuf1J2GEmLJPVJ6nvllVdGWEUzMysb6f3050bEDkknAvdJenaQZVWhLCotGBErgBUAPT09FZcxM7PajSjoR8SO5HGXpB9RSte8LGlSROyUNAnYlSzeD0xJrd4F7BjJ+7c1j5ZlZk0w7PSOpHdJOro8DXwEeAZYAyxMFlsI3JVMrwEWSDpS0jRgOrB+uO/f9sqjZfW+Cl/e2OzamFlBjKSlPxH4kaTy69waET+R9DiwWtJlwIvARQARsUnSamAz8CZwRUQcGFHtzcysJsMO+hHxf4E/q1C+Gzi7yjpLgaXDfU8zMxsZX5FrZlYgI+29Y7XwyVszazIH/UYqn7w1M2sSp3fMzArELf16c0rHzFqIg369OaVjZi3EQb8e3Lo3sxbloF8Pbt2bWYvyiVwzswJx0DczKxCnd/LiPL6ZtQEH/bw4j29mbcDpHTOzAnHQNzMrEAd9M7MCcU5/JHzy1qx4xk2F3nFvT7fZyHcND/qS5gPfAUYB34+IZY2uQ2588taseNJBvhz820hDg76kUcA/Ah+mNFD645LWRMTmRtZjRNy6N7M21uiW/mxgWzLUIpJuB86nNG5u86UDejXjprp1b2Yl6VTPwPIWTfsoIhr3ZtKFwPyI+PfJ888AH4yIKwcstwhYlDx9L7C1zlUbD/yuzu9RK9cpu1asl+uUTSvWCVqzXrXW6V9FxISBhY1u6atC2WF7nYhYAayof3VKJPVFRE+j3i8L1ym7VqyX65RNK9YJWrNeedWp0V02+4EpqeddwI4G18HMrLAaHfQfB6ZLmibpncACYE2D62BmVlgNTe9ExJuSrgR+SqnL5k0RsamRdaiiYamkGrhO2bVivVynbFqxTtCa9cqlTg09kWtmZs3l2zCYmRWIg76ZWYEUJuhLukjSJklvSara7UnSfElbJW2TtCRVfryk+yQ9lzwel0OdhnxNSe+VtCH19wdJX0rm9Up6KTXvvEbUKVluu6SNyfv21bp+3nWSNEXSzyVtSX7nL6bm5fY9Vds+UvMl6bpk/tOSTs+67khkqNenk/o8LemfJf1Zal7F37IBdTpT0qup3+VrWdetY53+U6o+z0g6IOn4ZF69vqebJO2S9EyV+fluUxFRiD/gFEoXej0E9FRZZhTwPPCvgXcCTwGnJvP+G7AkmV4C/EMOdarpNZP6/ZbSRRcAvcB/zPl7ylQnYDswfqSfKa86AZOA05Ppo4FfpX67XL6nwbaP1DLnAfdSuiZlDvBY1nXrXK8/B45Lpj9artdgv2UD6nQmcM9w1q1XnQYs/5fAg/X8npLX/TfA6cAzVebnuk0VpqUfEVsiYqgrew/eJiIi/h9Qvk0EyeOqZHoVcEEO1ar1Nc8Gno+IX+fw3nnVKe/1h/WaEbEzIp5MpvcCW4DJObx32mDbR7quN0fJOuBYSZMyrlu3ekXEP0fE75On6yhdI1NPI/m89fquan3di4HbcnjfQUXEL4B/GWSRXLepwgT9jCYDv0k97+ftwDExInZCKcAAJ+bwfrW+5gIO3wivTA75bsojlVJDnQL4maQnVLptRq3r16NOAEjqBj4APJYqzuN7Gmz7GGqZLOsOV62vfRmllmNZtd+yEXX6kKSnJN0r6bQa161XnZD0J8B84Iep4np8T1nkuk111P30Jd0PvLvCrP8SEXdleYkKZSPq0zpYnWp8nXcCfwVclSq+EfgmpTp+E/g28DcNqtPciNgh6UTgPknPJi2WYcnxezqK0j/qlyLiD0nxsL6nSi9foWzg9lFtmdy3rQzvefiC0l9QCvrzUsW5/pY11OlJSqnK15LzLP8bmJ5x3XrVqewvgUciIt0Cr8f3lEWu21RHBf2IOGeELzHYbSJeljQpInYmh1a7RlonSbW85keBJyPi5dRrH5yW9D3gnkbVKSJ2JI+7JP2I0qHmL2ji9yRpNKWAf0tE3Jl67WF9TxVkuY1ItWXemWHd4cp0exNJ7we+D3w0InaXywf5Letap9ROmYhYK+kGSeOzfp561CnlsKPqOn1PWeS6TTm9c6jBbhOxBliYTC8Eshw5DKWW1zwsv5gEwLJPABXP/uddJ0nvknR0eRr4SOq9m/I9SRLwA2BLRFwzYF5e31OW24isAS5NelzMAV5NUlL1vAXJkK8taSpwJ/CZiPhVqnyw37LedXp38rshaTaleLQ7y7r1qlNSl3HAvyW1ndXxe8oi320q7zPRrfpH6Z+9H9gHvAz8NCk/CVibWu48Sj0/nqeUFiqXnwA8ADyXPB6fQ50qvmaFOv0JpX+GcQPW/ydgI/B08mNPakSdKPUWeCr529QK3xOldEUk38WG5O+8vL+nStsH8Hng88m0KA0U9Hzynj2DrZvj9j1Uvb4P/D713fQN9Vs2oE5XJu/5FKWTy39e7+9qqDolzz8L3D5gvXp+T7cBO4H9lGLUZfXcpnwbBjOzAnF6x8ysQBz0zcwKxEHfzKxAHPTNzArEQd/MrEAc9M3MCsRB38ysQP4/TiZwAoHOHdcAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(azimuth, histtype = 'step', bins = 100, label ='truth') \n",
    "plt.hist(azimuth_pred, histtype = 'step', bins = 100, label = 'pred')\n",
    "plt.legend()\n",
    "\n",
    "print('correlation: ')\n",
    "cor = np.corrcoef(azimuth_pred,azimuth)\n",
    "print(cor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'azimuth prediction')"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "plt.hist2d(azimuth,azimuth_pred,bins=100);\n",
    "plt.text(-0.95,0.60,'Correlation=0.654',color='white',fontsize=20)\n",
    "plt.xlabel('azimuth truth')\n",
    "plt.ylabel('azimuth prediction')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "transformers = pd.read_pickle(\"transformers.pkl\")\n",
    "azimuth_transformer = transformers['truth']['azimuth']\n",
    "\n",
    "\n",
    "azimuth_pred_ns = azimuth_transformer.inverse_transform(np.array(azimuth_pred).reshape(1,-1))\n",
    "azimuth_pred_ns = np.concatenate(azimuth_pred_ns)\n",
    "azimuth_ns = azimuth_transformer.inverse_transform(np.array(azimuth).reshape(1,-1))\n",
    "azimuth_ns = np.concatenate(azimuth_ns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "correlation: \n",
      "[[1.         0.65400824]\n",
      " [0.65400824 1.        ]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,8))\n",
    "plt.hist(azimuth_ns, histtype = 'step', bins = 80, label ='Truth',color='green') \n",
    "plt.hist(azimuth_pred_ns, histtype = 'step', bins = 80, label = 'Prediction',color='deepskyblue')\n",
    "plt.xlabel('azimuth angle (radians)',fontsize=16)\n",
    "plt.ylabel('Counts',fontsize=16)\n",
    "plt.legend(fontsize=14)\n",
    "plt.xticks(fontsize=14)\n",
    "plt.yticks(fontsize=14)\n",
    "plt.legend(fontsize=14,loc=9)\n",
    "\n",
    "print('correlation: ')\n",
    "cor = np.corrcoef(azimuth_pred_ns,azimuth_ns)\n",
    "print(cor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1a7b521f160>]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,8))\n",
    "\n",
    "plt.hist2d(azimuth_ns,azimuth_pred_ns,bins=100);\n",
    "plt.text(0.1,5.0,'Correlation=0.654',color='white',fontsize=20)\n",
    "plt.xlabel('azimuth truth (radians)',fontsize=16)\n",
    "plt.ylabel('azimuth prediction (radians)',fontsize=16,color='deepskyblue')\n",
    "plt.xticks(fontsize=14)\n",
    "plt.yticks(fontsize=14)\n",
    "plt.plot([0.0,7.0],[0.0,7.0],'--',color='red')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plt.plot(pred[:,1])\n",
    "def PullFromDB(db_file,events,truths):#feats,truths):\n",
    "    ###\n",
    "    # HENTER ALLE EVENTS I events-variablen\n",
    "    #\n",
    "    \n",
    "    with sqlite3.connect(db_file) as con:                                           \n",
    "        #query = 'select %s from features WHERE event_no IN %s'%(feats,str(tuple(events)))                                        \n",
    "        #features = pd.read_sql(query, con)                       \n",
    "        query = 'select %s from truth WHERE event_no IN %s'%(truths,str(tuple(events)))                                              \n",
    "        truth = pd.read_sql(query, con)                             \n",
    "        \n",
    "    cursor = con.cursor()                                                       \n",
    "    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table';\")\n",
    "    \n",
    "    return truth#features,truth\n",
    "\n",
    "db_file = 'rasmus_classification_muon_3neutrino_3mio.db'\n",
    "dbfile = 'predictions.db'\n",
    "\n",
    "truths = str('energy_log10')\n",
    "energy_scaled = PullFromDB(db_file,event_no['event_no'],truths)\n",
    "energy_non_scaled = PullFromDB(dbfile,event_no['event_no'],truths)\n",
    "\n",
    "\n",
    "truths = str('azimuth')\n",
    "azimuth_scaled = PullFromDB(db_file,event_no['event_no'],truths)\n",
    "azimuth_non_scaled = PullFromDB(dbfile,event_no['event_no'],truths)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "abs_stop_pred = abs(azimuth_non_scaled['azimuth']-azimuth_pred_ns)\n",
    "#abs_stop_pred = stopped_non_scaled['stopped_muon']-pred[:,1]\n",
    "pr=abs_stop_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAj0AAAH1CAYAAAAdwk9FAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAABT40lEQVR4nO3de7xUVf3/8deHw0XkbohkfBVLRdQMPVQeUxuvZdY3L5Vpippmqfkrr2WakZbXIk27aGoiafmtzBTLuztvVIK3EEUrb3hBvICAB5HD5/fH2gPDsOec2efsOXN7Px+PeRxm7zVrr9kMzOes9VlrmbsjIiIi0uj6VLsBIiIiIr1BQY+IiIg0BQU9IiIi0hQU9IiIiEhTUNAjIiIiTUFBj4iIiDQFBT0iIiLSFKoW9JjZZDPzoscr1WqPiIiINLa+Vb7+XCBX8LyjSu0QERGRBlftoGeFu6t3R0RERCqu2jk97zezF83sGTP7nZm9v8rtERERkQZl1dp7y8z2AoYATwKjgNOBLYCt3P31hPJHAUfFT1sHDBjQW00VEZEm0Gf5cgBW9u9fldd35p133nF3r3ZHRd2rWtBTzMwGA/8FznX3KZ2VHTRokC9durR3Glanoigil8tVuxkNS/e38nSPK0v3N0H+fkRRj1+f9f01s7fdfVBmFTapmoka3X0J8DiwWbXbIiIiIo2nZoIeM1uHMLz1crXbIiIiIo2nmuv0/MjMPm5mm5jZR4E/AIOAqdVqk4iIiDSuak5ZHwP8FhgJLAD+Dmzv7s9VsU0iIiLSoKoW9Lj7F6t1bREREWk+NZPTIyIiIlJJ1V6RWaRuLVu2jAULFrBs2TJWrFhR7eY0nGHDhvHEE09UtQ39+vVj1KhRDB06tKrtEJFsKOgR6YaWlhaef/551l9/fUaPHk3fvn0xs2o3q6EsXryYIUOGVO367k57ezsvvvgigAIfkQag4S2Rblh33XUZM2YMI0aMoF+/fgp4GpCZse666/K+972PV199tdrNEZEMKOgR6YY+ffowcODAajdDesHAgQN59913q90MEcmAgh6RblLvTnPQ37NI41DQI1JLcrnV+/eIiEimFPSIiEh90i8JkpKCHhFZy5FHHomZccIJJ6R+7bPPPouZcdVVV2XfsC5ceOGFXH/99Wsdnzx5MmampQVEmpyCHhFZQ3t7O7///e8BuOaaa1IHCu9973uZMWMGe++9dyWa16lSQY+ICCjoEZEif/rTn3jrrbf41Kc+xauvvsott9yS6vUDBgxg++23Z/31169QC0VEukdBj4isYerUqYwYMYKrrrqKgQMHcvXVV686F0URZpb4OOyww4Dk4a3DDjuMMWPGMHPmTHbYYQcGDhzIuHHjuPnmmwGYMmUKY8eOZejQoXz2s59lwYIFq15bargs35YoigAYO3Yszz33HNdcc81abcp75pln2HvvvRk8eDAbb7wxZ555JitXrszs3olIbVPQI1JDZozZknN2PIgZi6pz/Zdeeok77riDAw44gPXXX5999tmHG2+8kTfffBOA7bbbjhkzZqzxOPvsswEYP358p3W/9dZbTJo0iSOPPJI//elPjBo1iv33358TTzyRu+++m5/97GdceOGF3H333Rx77LGp2/6nP/2J0aNH84lPfGJV27773e+uUWbfffdl11135YYbbmCfffbhe9/7HlOnTk19LRGpT9qGQqRGzFgEux06heUt/ej/KNz5IWgb1rttmDZtGitXrmTSpEkAHHroofz2t7/luuuu42tf+xpDhw5l++23X1X+6aef5oILLmD//ffnlFNO6bTuxYsX88tf/pKdd94ZgA033JAPfehDTJ8+nTlz5tDS0gLA7Nmzufjii7nssstStX3bbbdlwIABjBw5co02FjrxxBM5/PDDAdh999256667+O1vf7vqmIg0NvX0iHTTrFmz1nr0RLQQlrf0o6NPC8tXhue97eqrr2azzTajra0NCIHBhhtuuMYQV96bb77Jpz/9aTbddFOmTZvW5SJ+gwYNWhXwAGyxxRarrpEPePLHV6xYwSuvvJLFW1pDcXL11ltvzfPPP5/5dUSkNqmnR6QMxQHNk08+uepLOyu54dC/412Wu9O/X19ywzOtvksPPvggc+bM4Vvf+hYLFy5cdXy//fbjkksu4amnnmLzzTcHYMWKFXzuc59j2bJl/O1vfytrS47hw4ev8bx///4AjBgxIvH4smXLevBukq233nprPB8wYEBFriMitUk9PSI1om0Y3Dn1BM66+8qqDG3lc1vOO+88RowYsepxySWXAKzR23Psscfy4IMPMn36dEaPHl3Rdq2zzjoALF++fI3jr7/+ekWvKyKNRz09IjWkbd4c2ubNgR8c1avXXb58Ob/73e/46Ec/yrnn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\n",
      "text/plain": [
       "<Figure size 576x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "counts, edges = np.histogram(energy_non_scaled['energy_log10'],bins=80)\n",
    "accuracies = []\n",
    "errors = []\n",
    "nrn = []\n",
    "for a, b in zip(edges[:-1], edges[1:]):\n",
    "    fltr = (a <= energy_non_scaled['energy_log10']) & (energy_non_scaled['energy_log10'] <= b)\n",
    "    accuracies.append(np.mean(pr[fltr]))\n",
    "    errors.append(np.std(pr[fltr]) / np.sqrt(len(pr[fltr])))\n",
    "    #nrn.append(len(pr[fltr][pr[fltr]>0.5])/len(pr[fltr]))\n",
    "\n",
    "\n",
    "fig = plt.figure(figsize=(8,8))\n",
    "\n",
    "ax = fig.add_subplot(111)\n",
    "ax.errorbar((edges[:-1] + edges[1:])/2, accuracies, fmt='.',color='deepskyblue', label='Azimuth', yerr = errors, ecolor='red')\n",
    "#ax.plot((edges[:-1] + edges[1:])/2, nrn, label='Ratio of correctness')\n",
    "\n",
    "#ax.plot((edges[:-1] + edges[1:])/2,stest)\n",
    "ax.grid()\n",
    "ax.set_ylim(0,5)\n",
    "ax.set_xlabel(r'$log_{10}(Energy)$ [GeV]', fontsize=18)\n",
    "ax.set_ylabel('Mean Absolute Residuals',fontsize=18,color='deepskyblue')\n",
    "\n",
    "ax.legend(fontsize=16,loc=9)\n",
    "#ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\n",
    "ax.tick_params(axis='x',labelsize=14)\n",
    "ax.tick_params(axis='y',labelsize=14)\n",
    "\n",
    "\n",
    "ax2 = ax.twinx()\n",
    "ax2.hist(energy_non_scaled['energy_log10'],bins=80,alpha=0.2,color='k');\n",
    "ax2.set_ylabel('Counts', fontsize=20)\n",
    "ax2.tick_params(axis='y',labelsize=14)\n",
    "plt.savefig('areg2a')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-0.020005268412217066\n",
      "1.396209666877421\n"
     ]
    }
   ],
   "source": [
    "def calc_sigma(truth, pred, bins):\n",
    "    res = truth-pred\n",
    "    counts, edges = np.histogram(res,bins=bins)\n",
    "    mids = (edges[:-1] + edges[1:])/2\n",
    "    mean = np.average(mids, weights=counts)\n",
    "    var= np.average((mids-mean)**2, weights=counts)\n",
    "    sigma = np.sqrt(var)\n",
    "    return mean, sigma\n",
    "\n",
    "mean_pred, sigma_pred = calc_sigma(azimuth_ns,azimuth_pred_ns,80)\n",
    "print(mean_pred)\n",
    "print(sigma_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6.246179282211838\n",
      "3.4950545447955372\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.06283110227700033"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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3HvvvVx55ZVMnjyZe++9l1NOOSV27DfffDNDhgzhs5/97PrYzj777DZtvvSlL7Hffvvx97//nUMPPZRzzjmHa665JvZniUjt0ZYVInVm1lL49PhLWNXUi03+DXd/FMYN6N4Ypk+fzrp16zjuuOMAGD9+PDfccAM33ngj3/zmN9lss83Yfffd17d//vnnueiiizj88MP5wQ9+0O61ly9fzu9+9zv23ntvALbeems++tGPctttt/H000/T1NQEwNy5c7n88stZu3bt+mPl2G233ejduzdbbrllmxjznXHGGUycOBGA/fffn3vuuYcbbrhh/TERqV/q8RGpM9l3YFVTL9b2aGLVuvC6u1177bXssMMOjBs3DgjJwdZbb92m3JWzZMkSPv/5z7P99tszffr0Dhf669u37/qkB2DkyJHrPyM/wRk5ciRr1qzhtddeS+IrtVE44HrnnXfm5ZdfTvxz6klzc/P6x7x582hubq52SCKdoh4fkTqTGQibrF3NKnc26dWTzMDu/fzm5maefvppzjzzTN555531xw877DCuuOIKnnvuOXbccUcA1qxZwxFHHMHKlSu57777ytq+Y+DAgW1eb7LJJgBsvvnmRY+vXLmyC9+muC222KLN6969e1fkc2qVkhppZOrxEakz4wbA3deczs/u/WNVylzXX389AL/61a/YfPPN1z+uuOIKgDa9PqeccgqPPfYYt912G0OGDKloXJtuuikAq1atanP8rbfequjnikh9UY+PSB0a9+rTjHv1afj5Sd36uatWreKmm27ik5/8JBdccMFG50877TSmT5/Oz372MyZPnsxVV13FjBkz2GWXXSoe21ZbbUXv3r2ZO3dum+O5GWH5evfuTWtra8VjqjXFenLGjBlThUhEqkeJj4iU7bbbbuPtt9/m5JNPJlNkr7BvfOMbnHzyyZx//vn89Kc/5bjjjmOLLbbg4YcfXt9m0KBBfPjDH048NjPjqKOO4qqrrmLHHXdcPw0+N40930477cQDDzywvidqyy23ZPjw4YnHJCK1R4mPiJTtmmuuoX///nz5y18uev7oo4/m9NNP5/e//z3r1q1j2rRpG62rM378+IptU3HZZZexbt06Jk2axLp16zjyyCO5/PLL+fznP9+m3S9/+UtOPPFEjjzySFpbWysaU63TeB5JGyU+IlK2W265heXLl/O+972v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\n",
      "text/plain": [
       "<Figure size 576x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "abs_stop_pred = abs(azimuth_non_scaled['azimuth']-azimuth_pred)\n",
    "#abs_stop_pred = stopped_non_scaled['stopped_muon']-pred[:,1]\n",
    "pr=abs_stop_pred\n",
    "counts, edges = np.histogram(azimuth_non_scaled['azimuth'],bins=80)\n",
    "accuracies = []\n",
    "errors = []\n",
    "nrn = []\n",
    "for a, b in zip(edges[:-1], edges[1:]):\n",
    "    fltr = (a <= azimuth_non_scaled['azimuth']) & (azimuth_non_scaled['azimuth'] <= b)\n",
    "    accuracies.append(np.mean(pr[fltr]))\n",
    "    errors.append(np.std(pr[fltr]) / np.sqrt(len(pr[fltr])))\n",
    "    #nrn.append(len(pr[fltr][pr[fltr]>0.5])/len(pr[fltr]))\n",
    "\n",
    "\n",
    "fig = plt.figure(figsize=(8,8))\n",
    "\n",
    "ax = fig.add_subplot(111)\n",
    "ax.errorbar((edges[:-1] + edges[1:])/2, accuracies, label='Azimuth', fmt='.', color='deepskyblue', yerr = errors, ecolor='red')\n",
    "#ax.plot((edges[:-1] + edges[1:])/2, nrn, label='Ratio of correctness')\n",
    "\n",
    "#ax.plot((edges[:-1] + edges[1:])/2,stest)\n",
    "ax.grid()\n",
    "ax.set_ylim(0,10)\n",
    "ax.set_xlabel('Azimuth [radians]', fontsize=20)\n",
    "ax.legend(fontsize=16,loc=9)\n",
    "#ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\n",
    "ax.tick_params(axis='x',labelsize=14)\n",
    "ax.tick_params(axis='y',labelsize=14)\n",
    "ax.set_ylabel('Mean Absolute Residuals',fontsize=20,color='deepskyblue')\n",
    "\n",
    "\n",
    "ax2 = ax.twinx()\n",
    "ax2.hist(azimuth_non_scaled['azimuth'],bins=80,alpha=0.2,color='k');\n",
    "ax2.set_ylabel('Counts', fontsize=20)\n",
    "ax2.tick_params(axis='y',labelsize=14)\n",
    "\n",
    "print(max(accuracies))\n",
    "np.argmax(accuracies)\n",
    "qpr = (edges[:-1] + edges[1:])/2\n",
    "print(qpr[44])\n",
    "(abs(min(azimuth_non_scaled['azimuth'])-max(azimuth_non_scaled['azimuth'])))/100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.38309903036118464  lavest mean residual\n",
      "0\n",
      "2.7882046441792836  bin center\n",
      "0.06283110227700033  afvigelse\n"
     ]
    },
    {
     "data": {
      "image/png": "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twhZ96z7OKz7OPPet3W9IzGzWpq7HczdKY0NrOF/gl588UVGjhzJOeecw/rrr8+sWbM47bTT2G+//Zg2LQ1U/PjHP7781+1++9vfcsEFF7Dtttt2ev4nn3ySE044gZNOOonBgwfzne98h/3335/999+fJUuWMGXKFGbMmMEJJ5zA0KFDOfPMM3PFP23aNMaOHcuECRP42te+BrDSuKB58+ZxyCGHcOyxx3LKKafwq1/9iqOOOopRo0ax995757qW5eNEyLqn/X9TdRhwaGZ10v6ixLIBsA9pWc0aJUN77rkne+655/Lt3XffnS233JJx48bx0EMPsfPOO7PRRhux0UYbLa9z7733ctlll3Hcccdx0EEHdXr+1157jfvuu4/NN98cgGXLlnHAAQfw1FNPcfvttwPw0Y9+lLvvvptrrrkmdyLU3h03YsSIkl1z8+fP56KLLlqe9Oy5557cdttt/O53v3MiVGXuGjMzs/K0sPxFienvTGf6b6czfXr6PPTQQ1W99KJFizjttNPYZpttGDRoEAMGDGDcuHEAzJw5c5X6Tz/9NP/5n//JRz/6UX72s591ef6tt956eRIEsM022wAp+Sm0zTbb8Pzzz5PWIa2ctddee6WEZ80112Srrbbi2Wefreh1bFVuETIzs05Nnz49/WIYoHcg+sGA/jC6djGceOKJnH/++ZxyyinsvvvuDBkyhOeff54DDzyQt99+e6W68+bN4xOf+AQjR47kyiuvZI01uv4///rrr7/S9sCBAzssX7JkCUuXLqV//8p9hRZfB1IyVHxvVnlOhMzMrDw7AludDQu2hh8fmLZr5KqrruJLX/oSJ5988vKyBQsWrFJv6dKlfP7zn+eNN97gwQcfpKmpqapxrbXWWkBqsSr02muvVfW6VjlOhMzMGtTylp4Co0d30cwzeFb67HhglaIq7a233mLAgAErlf3qV79apd7xxx/P3XffzT333MOIESOqHtd73/teAGbMmMGnPvWp5eU33XTTKnUHDhxIW1tb1WOyfJwImZlZj7fvvvty+eWXs8MOO7Dlllvyxz/+kfvuu2+lOldddRXnnXceJ554Iu+88w7333//8n0jR46syuzNw4cPZ6+99uJ//ud/GDFiBEOHDuU3v/kN//73v1epu9122/HnP/+Zfffdl/XXX5+NN96YjTfeuOIxWT4eLG1mZsu1D34u/PQE559/Pvvvvz8nnXQSBx10EPPnz+d3v/vdSnX++c9/AnD66aczduzYlT4///nPqxbbb37zG3bddVe++c1vMmHCBDbZZJOVuvDaXXDBBTQ1NfHJT36SXXfdlcmTJ1ctJiufKj3yvSdramqKwomwKqGlpYXmBl6JudHvH/wMwM+gJ97/jBkzupw7p1tJzsSJ6WfRl/jChQtXer290cyfP58hQ4bU5dpd/V5LeisiqjtYqhdz15iZmVVEt8Yc9XLt97xw4cLlA7P7+j33NU6EzMysfH2wO6cREzhbwYmQmZlZGXrKeCmrLA+WNjMzs4blFiEzM7Mibv1pHG4RMjProxrpreBG5d/j1edEyMysDxowYIBnMW4AbW1tq8y4bfm4a8ysgN8esb5i6NChvPDCC4wYMYJBgwYhqd4hNYxa/DsSEbS1tfHCCy/w7ne/u6LnbjROhMzM+qB11lkHgBdffJHFixeXrPPUU09V7HrvvPMOa6655irla6+9dsWuUS2VeA4d3X+7ajyHAQMG8O53v3v577V1jxMhM7M+ap111un0S/Ktt96q2LVaW1vZZpttVinvanbrnqASz6Gj+2/XG55Do3IiZNZHuZvPzKxrHixtZmZmDatHtAhJOho4ARgOPAYcGxH3dFL/c8D3ga2BV4ALIuKntYjVzMzK19F8PG6dtJ6i7omQpIOAc4Gjgb9mP2+WtF1EPFui/seAK4FvArcA2wKXSWqLiAtqF7n1Ju4mMjOzUnpC19jxwJSIuCwiZkTEMcBLwFEd1D8UuCEiLoqIWRHxZ+B04Lvy+6FmZmaWg+o5K6WkgcBbwMERcU1B+YXA9hGxV4ljrgUWRcTBBWVfAS4DNouIp4vqTwQmAvTv33/01KlTK3oPCxYsYPDgwRU9Z2/SE+9/5syZFT3fqFGjOt3fE58BlH4OXd1Ld/XUZ1ArveH+K/33olhbWxuDBg0qu361/ix2RyWeTVf3X8/73Xvvvd+KiKa6BdDD1btrbEOgHzC7qHw28OEOjrkVOFfSR4DbgS2Bb2f7hgNPF1aOiMnAZICmpqZobm6uRNzLtbS0UOlz9jjt99fSssqunnj/Q4YMqej5uupC64nPAEo/h2p1B/bUZ1ArveH+K/33olhraytjxowpu35P6pquxLPp6v570v3ayuqdCLUrbpZSibJ2lwFbANcBA4B5pDFGk4ClVYrPrE/wWCmz+vDfvZ6r3onQq6TkZVhR+VBWbSUCIFJf3nclfT877hVgn2z309UJs8HN3Q7m7gTTgLH1DsbMrPu8qrwVq2siFBGLJE0HxgPXFOwaD1zbxbFLgRcAJB0MTIuIOdWKtWFNAx45C5YNSOnmHTgZalB+DdrM+qJ6twgBnAVcIelB4F7gSGBj4BIASacDu0XEPtn2hsBngRZgTeDwbHuVgdVWAS2kJIh+sCjbbrBEqK8nAG6yt77KrT9WjronQhFxtaQNgJNJg50fBfaLiGeyKsNJY4IKfQn4KWks0TSgOSIerFHIjaUZWGMxLAsY2D9tm1mP5gTArHx1T4QAIuIi4KIO9k0o2n6VhmuTqL4OWz3GjoYdj09jhH470U++B6jFl5y/SM2sUfSECRWtp1v3cdjkSidBZmbW5/SIFiHr4UrMH2RmZtYXuEXIzMzMGpZbhMys4fhNOTNr50Soj/M/+GZmZh1zImS9hpM6MzOrNI8RMjMzs4blRMjMzMwalhMhMzMza1geI2S9mmdAtkopdwya/8xVhsf8WU/hRKiX8j8iZmZmq89dY2ZmZtawnAiZmZlZw3LXmHXKXXBmPZvHLJmtHrcImZmZWcNyi1Af4v8Z1lb78545cyZDhgwBGrO1bPr06Ss9A2jM52BmvZNbhMzMzKxhuUWoAbnlyKyyilvE2rllzKznc4uQmZmZNSwnQmZmZg1M0omS/iZpnqRXJN0gafuiOlMkRdHn/qI6a0o6X9KrkhZKul7SyKI660u6QtLc7HOFpPVqcJsdcteYmVkH3I1sDaIZuAj4GyDgR8DtkraLiNcL6t0OHFqwvajoPOcABwAHA68BZwE3ShodEUuzOlcCmwAfAwL4OXAF8MkK3k8uToQst8Ivh/axER4LYWaWT0eJdq3/PY2IjxZuSzoUmAt8ELihYNc7EfFyqXNIWhc4Ajg8IqYWnOcZ4MPArZK2BfYF9oiI+7I6XwPukTQqImZW9s7K40TIzPo0t+qY5TaENHTmjaLyPSTNAd4E7gJOiog52b7RwADgtvbKEfGcpBnA7sCtwFhgAXBfwTnvBRZmdZwIWWn+h9ysPP67YlZSf0mtBduTI2JyJ/XPBR4GphWU3QL8EXgK2BT4MXBn1u31DjAMWAq8WnSu2dk+sp+vRES074yIyJKrYdSJEyEzM7O+bUlEjCmnoqSzgD1I3Vft43qIiKsKqv1D0nRSt9fHSQlSh6ckjQVafqoy6tSU3xozMzMzJJ1NGuj8oYiY1VndiHgReB7YKit6GegHbFhUdSipVai9zlBJKrimgI0K6tScW4SsR3IXh5nl4X8zVo+kc4HPA80R8c8y6m8IjABeyoqmA4uB8aQ3w8hend+WFWOCpgGDSWOF2svGAk2sPG6oppwImZmZNTBJF5Jei/8U8Iak9vE6CyJigaTBwCTgWlLisylwOjAH+BNARMyV9Avgp9mYn/bX5x8hvXZPRMyQdAtwqaSvkrrELgVurNcbY+BEyMyqoNT/zj3FglmPdXT2846i8lNJCdBSYAfgS8B6pGToL8DnImJ+Qf3jgCXA1cCg7HxfKhxrBHwBOI8Vb5ddD3yjQvfRLU6ErCL8xWdm1jtFhLrY3wZ8tLM6Wb23gWOyT0d1Xge+mDfGanIiZGZWJf4PglnP57fGbGUTJ6aPmZlZA+gRiZCkoyU9JeltSdMljeui/kclTZM0P1vc7TpJW9cqXjMzM+sb6p4ISTqINIvlacDOpFfobpa0SQf1NwOuA+7J6n+YNCjrppoE3Nct2Bxe3jeN8zczM+vj6p4IAccDUyLisoiYERHHkEakH9VB/fb1TE6MiCcj4mHSa3xbZPMaWHc9AjxxHLx4QHr6TobMzKyPq+tgaUkDSYnNz4p23UZagK2UVtKkTV+R9HNgbeAw4G8RUbzGieUxHYj+QL/0hKcDO9Y3pN7Gg2NrxxPomVkl1PutsQ1JU3IXT609m9TltYqIeFrSeOAa4EJSq9ZDwMdK1Zc0EZgI0L9/f1paWioSeLsFCxZU/JzFZs6szTxTg5sGM4pNgWBZP/FE0xMsaF3Q6TFtbW20traW3Dd//vyS5eWo1T1XQmfPAFbvOZTSE59NV88A+vZzKOf+2/XV55DnGXSkN/+bUYn7b1fpPyPWuXonQu2KF1vrcAG2bMbLXwC/Bn4HDAF+BPxe0ociYtlKJ04r7E4GaGpqiubm5ooG3tLSQqXPWWzIkCFVPf9yY4D/PQMWbE2/Hx/INjtu0+Uhra2tjBlTei2/1WkJqdk9V0BnzwAq3yLUE59NV88A+vZzKOf+2/XV55DnGXSkN/+bUYn7b+dW5NqqdyL0KmnGymFF5YWLtBX7OrAwIr7TXiDpi8BzpO60v1YhzsYxeFb67HhgvSMx65PcfWrWs9Q1EYqIRZKmkxZpu6Zg13jSmialrE1Kngq1b/eEwd+92+TJ9Y7AzMysZurdIgRpUbYrJD0I3AscCWwMXAIg6XRgt4jYJ6v/Z+A4ST8krXA7hPTq/XOk4b29mgeAWiH/eTAzq666J0IRcbWkDYCTgeHAo8B+EfFMVmU4sEVB/TslHQJ8BzgBaAPuB/aNiIU1Dd7MzMx6tbonQgARcRFwUQf7JpQouwq4qsphlWXmzJmrDNJzf7+ZmVnvUJkxNWJ9RFNFzmVmZmZWI+UnQmIfxJmI9QvKhiLuIr399TrirMqHaGZmZlYdeVqEjgEOJHijoOxnwDjgSeA14FuIz1UwPjMzM7OqyZMIvZ/COXrEIOAzwFSCUcAo0ptbR1YyQDMzM7NqyZMIDQVeLNj+D2AtYAoAwXzgRlJCZGZmZtbj5UmE3gEGFWyPIy2DcXdB2TzgXRWIy8zMzKzq8rw+/xTwoYLtTwNPELxQUPYe0sBpMzOrME+waVZ5eVqELgd2QDyAuAfYgTSzc6FdgJ6xFLKZmZlZF/K0CF0MfAA4iLQ6/A3Afy/fK3YDtiWtCG9mZmbW45WfCAWLgUMQRwKRDY4uNAvYGXi6YtGZmZmZVVH+JTaCeR2Uv4rHB5mZWTeVGgPlJYus2iqzxIaZmZlZL9Rxi5CY1c1zBrFitXgzM/D/9s2sZ+qsa2wN0jxBeambsZiZmZnVVMeJULBp7cIwMzMzqz2PETIzM7OGlf+tMasYzxJrZmZWX/kTIbEmsCswAlizZJ3g16sVlZmZmVkN5EuExJeBM4H1O6yRBlg7EeoNJk5MPydPrm8cZmZmdVL+GCGxL/Bz4CXg/5GSnuuAk4Cp2fY1wJcrHqWZmZlZFeQZLP1t4DVgd4Kzs7KHCc4g2Bf4KnAg8O8Kx2jVsmBzeHlfeKTegZiZmdVHnkRoF+CGojXGVhwf/AK4l9RCZD3dI8ATx8GLB8BROBkyM7OGlGeMUBOpW6zd28A6RXVacddY7zAdiP5AP1icbe9Yp1D89pyZmdVJnhahl4GNCrZfAkYV1VkX6Le6QVkNjAa0BFgCA7JtMzOzBpMnEXqMlROfe4B9EOMAENsDn8vqWU+3I7DV2bDx9XAxdWsNMjMzq6c8idDNwAcRG2fbZwJLgRbEK8DfgSHAjysbolXN4Fkw7BYnQWZm1rDyjBG6lPR6/BsABI8j9gFOBrYgjQ86h+DWSgdpVeL5g6yX8DgyM6uW8hOhYDEwu6jsfuATlQ3JzMzMrDa86KqZmZk1LCdCZmZm1rDK7xoTy0jriHUlCK9qb2ZmZj1fnoTlbkonQusBWwODSG+OvbnaUVmfUGqA6+jRnrDIzMx6jjyDpZs73CeGAGcDu5PWGzMzMzPr8SozRiitPzYRWAL8JO/hko6W9JSktyVNlzSuk7qTJEUHn6GrcRdmZmbWYCo3lidYhvgL8Fng6HIPk3QQcG52zF+znzdL2i4ini1xyM+AS4rKrgIiIuZ0K3YzszpyN7JZ/VT6rbG1gPVzHnM8MCUiLouIGRFxDGkds6NKVY6IBRHxcvuHtFLWOOCy1QnczMzMGk/lWoTENqTWoCfLPkQaSFru82dFu24jjTcqxxGkAdrXdnCNiaRuO/r3709LS0u54ZWlra2N1tbWlcqKtwFGjSpenxZmzpxZ0VjqodT9d2b+/PmrlPX259DVMyh1z+XqLc8m75+DduU+m57+HLp7/53pbX9XqvEMoPf8Gank/a/OvxmWX57X53/ZyTneA3yQtPL8t3Ncf8PsmNlF5bOBD3cZkrQG8GXg1xHxTqk6ETEZmAzQ1NQUzc3NOcLr2syZMxkzZkyX9Uo1cw8ZMqSisdRDa2trWfffri8+h66ewep0cfSWZ5P3z0G7cp9NT38O3b3/zvS2vyvVeAZQ+jmU6kqsxrXzqOT9u1u0tvK0CE3oYv8/gZ8S/KobcRS/lq8SZaXsR0rCft6Na5qZmVmDy5MIbdZB+TLgDYIF3bj+q6QV7IcVlQ9l1VaiUr4K3BcRj3Xj2mZmZtbg8swj9EylLx4RiyRNB8aTVrZvN54Oxvy0k7Qx8HHgK5WOy8zMzBpDT1hr7CxggqSvSNpW0rnAxmSvyEs6XdIdJY77MrAQ+H3tQjUzM7O+pOMWIbFnt88a3F121YirJW0AnAwMBx4F9ouI9hao4cAWK4UmifS22G8j4q1ux2lmZmYNrbOusRbKG7BcSr88lSPiIuCiDvZNKFEWdDxmqUcq9ZaDmZmZ1VdnidCPWDUR+g9gX+DfpFmgXyYNdN6D1GpzM/Bg5cM0MzMzq7yOE6Fg0krb4gPAicC3gAsJlhXsWwM4BjiDlECZleSWMTMz60nyvD7/X8DtBOevsiclRecixpMSoY9WJjyz3s/rSFl3+D8NZrWRJxHaDUokQSv7O/CN7odj1rj8xWdmVnt5Xp8XRW9vlbDlasRiZg1m+vTpq3zMrLYknSjpb5LmSXpF0g2Sti+qI0mTJL0oqU1Si6T3FdVZU9L5kl6VtFDS9ZJGFtVZX9IVkuZmnyskrVeD2+xQnkToPuDTiE+U3Cv2Bw4E7q1AXGZmZlYbzaQ3t3cHPgQsAW6X9K6COt8hrSV6DLArMAeYKqlwAbxzgE8DBwPjgHWAGyUVvkl+JbAL8DHSy1e7AFdU/I5yyNM1dhJwN3Ad4q7s17OBdwN7AXsCbVk9MzMz6wUiYqVxvZIOBeaSFlO/IZu771jgjIi4NqtzGCkZOgS4VNK6pPn9Do+IqQXneYa0iPqtkrYlJT97RMR9WZ2vAfdIGhURM6t+syXkWWJjejYY+pek7LGZ9Hq9shozgSMIHqpsiGZmZlZDQ0g9Rm9k25uRpsq5rb1CRLRJupvUinQpMBoYUFTnOUkzsjq3AmOBBaQepnb3klaJ2J2UR9RcnhYhCO4DtkHsTmrOWpeUNf5fts/MzKxiPG6sIvpLai3YnhwRkzupfy7wMDAt225fGL14MfTZwIiCOktJi6kX1xlWUOeVbFJkIE2QLGkOqy6+XjP5EqF2Kelx4tObTJyYfk7u7M++mZn1QUsiYkw5FSWdRZokeY+IWFq0u3iSZZUoW+WURXVK1S/nPFXTExZdtVpYsDm8vC88Uu9AzMysJ5J0Nmmg84ciYlbBrpezn8WtNkNZ0Ur0Mml5rQ27qDM0G3PUfk0BG7Fqa1PNdLbo6imkDO1Cgtez7XIEwX9VIjirkEeAJ46D6A9HARcDO9Y5JjMz6zEknQt8HmiOiH8W7X6KlMSMB/6W1V+L9GbYCVmd6cDirM6VWZ2RwLas6EGaBgwmjRVqLxsLNFHHXqbOusYmkRKhq4HXs+1yBDgR6lGmk5Ig+qU/ptNxImRmZgBIuhA4FPgU8Iak9pafBRGxIBvHcw5wkqR/Av8CTiYNfL4SICLmSvoF8NNszM9rwFmk/4rfntWZIekW0ltmXyV1iV0K3FivN8ag80Ro7+zns0Xb1tuMBrQEImBA/7Rt1kN5cKxZzR2d/byjqPxUVjSCnAkMAi4E1gceAD4SEfML6h9HmoPo6qzuHcCXisYafQE4jxVvl11PnVek6GzR1bs63bbeY0dgq7Nhwdbw4wPdGmRmZstFhMqoE6SkaFIndd4mTbh4TCd1Xge+mDvIKvJg6UYxeBYMu8VJkJmZWYHyX58XmwLbAXcRLMzK+gM/IPUrLgR+SvCnSgdpFeDX5s3MzFaRZx6hHwL7k5bUaHcyKRFq93vEOIL7KxGcmZmZWTXl6RobC9xBsAQAsQZpgNU/gU2A3UitQsdVOEYzMzOzqsiTCL2btHhau51IEyddSPA8QStwHWlVWjMzM7MeL08iNICVp8D+YLZ9Z0HZ88DwCsRlZmZmVnV5EqHnWfmdo/2AVwlmFJQNBeZVIjAzMzOzasszWPpG4DjEz4C3SdNo/6qozjas3H1mZmZm1mPlSYTOJL0mf3y2/QLpTbJEvBfYHTi7QrGZmZmZVVX5iVAwB7EDsE9WchdB4dTag0lJ0q2VC8/MzMysevK0CEHQRuoiK7XvMeCx1Q/JzMzMrDbyJULtxDbAtsBggisqGpGZmZlZjeRba0zshGgltfz8AZhSsG8vxFuIT1YyQDMzM7NqKT8RElsDLcAo4Fzg5qIadwOvA5+pUGxmZmZmVZWnReiHwEBgN4Ljgb+ttDcIYBqeWdrMzMx6iTyJ0D7AH4smUCz2LLDx6oVkZmZmVht5EqH1SLNLd3W+gd2OxszMzKyG8rw1NgfYsos67wOeyxuEpKOBE0jrlD0GHBsR93RSX8C3gCOBzUhjky6PiO/lvbZZPUyfPr3eIZiZGflahO4EPokYVXKv2JXUfZZrQkVJB5EGX58G7AzcB9wsaZNODvsf4Gjgu6TX+PcjDdY2MzMzK1ueROh0YAlwN+Io2scCifdl2zcA84Gf5YzheGBKRFwWETMi4hjgJeCoUpUljQKOAQ6IiOsiYlZEPBQRN+W8bt81cWL6mJmZWafKT4SCmcCnSWOALgC+Agh4BLgwKz+Q4NlyTylpIDAauK1o122kdctKOQCYBewraZakpyVdLmlo2fdiZmZmRv4lNm5BbAYcBnwA2ACYC9wP/Irg9ZzX3xDoB8wuKp8NfLiDYzYH3gt8HpgABKkV6gZJYyNiWWFlSROBiQD9+/enpaUlZ4ida2tro7W1taLnXF3bzhlGv7btePr3/2TB5guqeq2eeP+15mfgZ9Do9w9+BpW8//nz53ddySom/xIbwZukMT3nltwvtid4NPdZi8+yalm7NYA1gUMj4l8Akg4FZpLmMHpgpRNHTAYmAzQ1NUVzc3PO0Do3c+ZMxowZU9FzrpZHgBd2gOjPNuf2g4uBHat3udbW1p51/3XgZ+Bn0Oj3D34Glbz/0aNHV+Q8Vp58S2x0RmyB+C3wUI6jXgWWAsOKyoeyaitRu5eAJe1JUOYJ0vilzgZYN4bpQPQH+sHibNvMzMxKKi8REuMQxyOOQexctG8Y4lLgceBgOk5gVhERi0hf1eOLdo0nvT1Wyr1Af0lbFJRtTmrdeqbca/dZowEtAZbAgGzbzMzMSuq8a0z0B64FPlFU/lOC7yG+CFwEDCYlQGcAl+SM4SzgCkkPkpKcI0lvpF0CIOl0YLeI2Cerfzvwf8AvJR2blZ1D6hJr3A7qdjsCW50NC7aGHx9Y1W4xMzOz3q6rMUJfBz4JLATuIo3d2Qs4AbEQmER6Zf57wPkEbXkDiIirJW0AnEyaUPFRYL+IaG/dGQ5sUVB/maRPAOeR5g5qA6YCxxcPlG5YV3peSTMzs3J0lQh9npTo7EwwCyCbULGVlAT9HdiP4OXVCSIiLiK1LJXaN6FE2UvAZ1fnmmZmZmZdjRHalrTQ6qzlJWk+oT9mW0eubhJkZmZmVi9dJUJDoOQEie3dVg9XNBozMzOzGuoqERLp9fZiqSxYVOmAzMzMzGqlnAkV10OrzM+zHgDiPaRkaWU5ltkwMzMzq5dyEqFvZZ9Sni5RFmWe18zMzKyuukpYnqXjpS7MzMzMerXOE6Fg09qEYWZmZtYxSTsDY4HfRsTcrKyJNP3OAcBbwH9HROm1UDtQubXGzMzMzKrnu8BJ7UlQ5nTgUFI+swFwlqSP5DmpEyEzMzPrDcYALe0bkgYAhwEPkhZr34y0mPs385zUiZCZmZn1BkOB5wq2x5DmO7w0It6OiBeB68i5yqYTITMzM+sNit9K3yMru6ug7BVgozwndSJkZmZmvcGzwAcKtg8Ano+IWQVlGwNv5DmpEyEzMzPrDX4P7C7pD5J+Q3qD7A9FdbYH/p3npJ740MzMzHqDs4F9gQOz7YeBH7XvlLQdMBo4Lc9JnQiZmZlZjxcRC4APSto+K3o8IpYVVHkL+E+gNc958ydCYiPg08C2QBPBVwrKNwP+QdCW+7xmZmZmHZC0CfBmRDxaan9EPC3pNWD9POfNN0ZIHEFaX+xC4Bjg8IK97wamAYfkOqeZmZlZ154Cju2izjezemUrPxES44HJwL9ITU8Xr7Q/eBR4DPhUngDMzMzMyqBqnDRP19h3gZeAvQjmIXYuUecR0ihuMzMzs1p7N7AwzwF5EqExwFUE8zqp8zwwLE8AtpomTkw/J0+ubxxmZmYVJulLRUU7lSgD6AdsQlp37B95rpEnERpI11nWesDSPAHYalqwOSzYOrXF5ZpU3MzMrMebQpo9muznAdmnWHu32VvAqXkukCcRepr0fn5n/gOYmScAWw2PAE8cB9EfjiKN2nIyZGZmfUf7S1kCfgn8L2k9sWJLgdeAaRHxZp4L5EmErgO+g/gswTWr7BWHk76GT8oTgK2G6aQkiH6wONt2ImRmZn1ERFze/mtJhwH/GxG/ruQ18iRCZwKfB36H+AywboqMbwDjSDM9PgGcX8kArROjAS2BCBjQv+v2OjMzs14qIvauxnnLT4SCNxB7Ab8GPluw57zs5z3AIUS+0dq2GnYEtjo7jRH68YFuDTIzM8sp38zSwbNAM2JH0mvyGwBzgfsJplc+POvS4Fnps+OBXdc1MzPrxSTtBZwA7EaaQbrUfIgREWXnN91bayx4hDRU1+rNr82bmVkDkPRx0mDpfsCzpJezlqzuectPhMQs4BxieVdYqTpfB75NsPnqBmZmZmZWYBLp1aCPR8RtlTppnrXGNiXNE9SZ9YD3djMWMzMzs45sD1xdySQI8i662rXBwKIKn9PMzMxsAfB6pU/aedeY2KSoZL0SZbBiauvPALMqE5qZmZnZcndQhfVMuxoj9DQrprYG+Fb26YiA41czJjMzM7Ni3wUelHQy8JOIiK4OKEdXidCvSYmQgC+R3hR7uES99qmt7yCoaN+dmZmZGfBD4DHSWmJflvQw8GaJehERR5R70s4ToWDC8l+LLwF/IvhRuScvl6SjSfMCDCfd5LERcU8HdTcFniqx62MRcUulY+tRvNK8mZlVmKQ9gf9HWp9gY+DwiJhSsH8KcFjRYQ9ExAcK6qwJ/Aw4GBhE6sY6OiKeL6izPmkS5v2zouuBY3KsDTah4NebZp9SAqhQIrTyaSs9sBoASQcB5wJHA3/Nft4sabuIeLaTQ/cF/l6wXfEBVD2OV5o3M7PKGww8SuoF6mgdr9uBQwu2i1+MOoe0KvzBpB6is4AbJY2OiKVZnStJ44k/RkpWfg5cAXyyzDg3K7NeLt2bULGyjgemRMRl2fYxkvYlrad+YifHvRYRL1c9up7CK82bmVkVRMRNwE2wvPWnlHc6+s6VtC6pBebwiJialR0KPAN8GLhV0rakBow9IuK+rM7XgHskjYqImWXE+UyuGytTngkVf1lmzSDKa5KSNJDUFPezol23Abt3cfgfJa1FWuj17Ij4QwfXmAhMBOjfvz8tLS3lhFa2trY2WltbK3rOUobdMowRMQzRj2WLlvHi9S/y8qL654G1uv+ezM/Az6DR7x/8DCp5//Pnz6/IeSpsD0lzSGNy7gJOiog52b7RwABYMUY4Ip6TNIP0XX4r6W2vBcB9Bee8F1iY1ekyEaqWPC1CE7rY3z6oOk/f3IakV+9nF5XPJmWRpSwg9WXeS5pae3/gakmHRcRvVgkqYjIwGaCpqSmam5vLDK08M2fOZMyYMRU9Z0kDgevegYA1BvZn5P4jGbnjyOpftwutra21uf8ezM/Az6DR7x/8DCp5/6NHj67IeQr0l1SYpU3OvhvLdQvwR9L43E2BHwN3Zt1e7wDDSC9NvVp03OxsH9nPVwrf9IqIyJKrYZRBUqnpe0rqYmjNSvIkQh31za0H7Ar8gJTpfS/HOdsVvwKnEmWpYsSrwP8UFLVK2hD4DrBKItRneKV5MzPrniUR0e0sLSKuKtj8h6TppG6vj5MSpI4Uf5eX+l7v8Pu+hKfLrBvkyG/yDJbuqG/uGeDviFtJI1luB35R5llfJWWRxdngUFZtJerMA8DhOer3Tld2J8c0MzOrnIh4UdLzwFZZ0cuk3p0NgVcKqg4F7i6oM1SS2luFJAnYiPK/79un9Cm2HrATaYmvFugwXympcoOlg+cQN5AmXCwrEYqIRVlmOR64pmDXeODaHFffCXgpR30zMzPrhqwXZgQrvnenkxZDHU96MwxJI4FtWTEmaBrp7bSxBWVjgSZWHjfUoYiY0ElMa5B6po5k1Vf9O1Xpt8ZmsyJDLNdZwBWSHiSN+zmSNI/BJQCSTgd2i4h9su3DSA/8IWAZ6bW7r5NmnDQzM7McJA0Gtsw21wA2kbQTaVqa10mrvl9LSnw2BU4H5gB/AoiIuZJ+Afw0G/PT/vp8ey8RETFD0i3ApZK+SuoSuxS4sZw3xroSEcuAU7O3zs8AvlDusZVLhEQ/4EPA3DyHRcTVkjYATiZNqPgosF/Ba3LDgS2KDjuZ1AS2FPgX8OVSA6XNzMysS2OAvxRsn5p9LidN2LIDaXWJ9UjJ0F+Az0VE4ettx5FeYLqaFRMqfqlgDiFIycl5rHi77HrgGxW+l/uyWMuW5/X5PTs5x3tIY3R2Ik2QlEtEXARc1MG+CUXbl5N+c8zMzGw1RUQLqYWmIx8t4xxvA8dkn47qvA58MW98Ob2L1N1WtjwtQi10PlpbpEFRJ+QJwMzMzGx1SfowcBCpZ6lseRKhH1E6EVoGvAE8SPBgnotbJ7yumJmZ2XKS7uxgV3vPVPs8Q7nWRM3z+vykPCe21eR1xczMzAo1d1AepAaZW4GfRURHCVNJPWGtMSvmdcXMzMxWEhFVWfy9Kie11TSdlATRL00UML3O8ZiZmfVRHbcIiVndPGcQq7zubnmMBrQEImBA/7RtZmZmy0laB1gXmBsR87p7ns5ahNYgvQmW9+NWptXVvq7Yxte7W8zMzCwjqZ+k70l6kjQu6GngDUlPZuW5h/x0fECwaXcDtQrwumJmZmbLSRoI3ALsRRog/RxpgsfhpBmvfwLsK+kjEbGo3PO69cbMzMx6g+NJb479Gdg2IjaNiLERsSkwCrgBGJfVK1v3EyGxDuI9iHW6fQ4zMzOz8hxCmizxUxHxROGOiPg3cCDwGDnWGYO8iZDoh/geYqW+OcSTWblfxzczM7Nq2BK4OVtgdRVZ+c2suj5pp/KsNVZW3xziIwRl982ZmZmZlWERMLiLOk2kiWfKlqdFaKW+OYJNCcZmg6q73TdnZmZmVoZHgM9I2qjUTkkbAp8B/p7npHkSoeV9cwQr9c0RdLtvzszMzKwMFwAbAQ9KOkLS5pIGSdpM0uHAA9n+C/KcNM+Yni2B8wlK9s0RLEPcDByTJwAzMzOzrkTE7yXtBHwPKLUiuYAzI+L3ec6bJxGqSt+cmZmZWTki4vuSrgeOAHYmm1kaeAj4ZURMy3vOPInQI8BnEJMIXlllr+hW31zDO+SMtMr8jw/0DNJmZmZdiIj7gfsrdb48Y4SW980hjkBsjhiE2AzR7b65hta+yvyLB6RV5h+pd0BmZmY9h6Q1JT0o6Q5JAzqpNzCrc39n9Uopv0Uo+D1iJ7romyPI1TfX0EqtMu9WITMzs3ZfIC09/smI6HDoTUQskvRT4KbsmCnlXiDfBIjB9xEd9s0R5O6ba2heZd7MzKwzBwKzIuKmripGxC2SngA+S9USIYCgon1zDa19lXmPETIzMytlZ1IrT7nuBvbLcwEviVFvXmXezMysIxsCs3PUnw1skOcC5Q+WFjsjjkasW1DWhLgc8SbiRcS38lzczMzMrBNtdD11T6HBwNt5LpDnrbHvAicRzC0oOx04NDvPBsBZiI/kCcDMzMysA88Bu+aoPwZ4Ns8F8iRCY4CW5VtiAHAY8CAwFNgMeBX4Zp4AzMzMzDrQAnxA0piuKkoaDewO/CXPBfIkQkNJmVm7McAQ4FKCtwleBK7DQ37NzMysMi4AArhG0rYdVZK0DXANsBS4KM8F8gyWjqL6e2RldxWUvUKaVNHMzMxstUTETEk/AiYBD0n6A3An8DwpBxkJ7AN8GlgTOCUiZua5Rp5E6FngAwXbBwDPE8wqKNsYeCNPAA1l4sT0c3Kp+SjNzMysWET8SNIS4IfAIcDBRVVEmpb4pIg4Pe/58yRCvwdORfyBNCJ7LHBOUZ3tgX/nDaJhLNg8zRn0CO5ANDMzK1NEnCbpt8CXgQ8Cw0kJ0IvAX4FfRcQz3Tl3nkTobGBf0iyPAA8DP1q+V2xHmhv5tO4E0ue1rysW/dO6YhfjZMjMzKxMWaLzw0qfN89aYwuADyK2z0oeJ1hWUOMt4D+B1sqF14d4XTEzM7MepztLbDzaQfnTwNOrFU1f5nXFzMzMepzuLbEhxlG86GpwT3eDkHQ0cAKpz+8x4NiI6PJ8krYC/g9QROSZebL2dgR+sWZqCRqNW4PMzMx6gHyJkPgg8Etgy+Ul6fU1EE8ARxDcm+uU0kHAucDRpAFPRwM3S9ouIjqcHVLSQOAq0gJre+W6j3rZESdAZmZmPUj5iZAYDUwF1iLNHdQCvAwMA/YG9gRuQ4wj+L8cMRwPTImIy7LtYyTtSxpSfGInx/03aQjyXfSWRMjMzMx6lDwzS/+ElDgdQLA3wakEl2Y/m0kDpQdm9cqSteqMBm4r2nUbaZrsjo77OPAJevJyHoecAfv/MaVqZmZm1iMpIsqsyTzgJoLPd1Ln98BHiYIV6js7pbQx8AKwV0TcXVB+CvCFiBhV4pjhpJE2B0bE/ZImABd0NEZI0kRgIkD//v1HT506tZzQyvbwww8zaNCglcoGzxrMqDM3BfqzbIB44rgnWLD5gopet6doa2tb5f4bjZ+Bn0Gj3z/4GVTy/keNWuWrb7Xsvffeb0VEU0VP2ofkGSO0DHiyizpPQLdWny/OxlSirN1vgIsj4v6yThwxGZgM0NTUFM3Nzd0Ir2MzZ85kzJiiteD+AWm5k370WwrbLNwmrczWB7W2tq56/w3Gz8DPoNHvH/wMKnn/o0f7teJaytM11gq8v4s67yetRl+uV0kZw7Ci8qHA7A6O+RDwQ0lLsim3fwE0ZdsTc1y7etpflWcJDMCvypuZmfVQeVqETgZaEEcRXLzKXvF10sJnzeWeMCIWSZoOjCetGttuPHBtB4ftULR9AHASsBupm63+/Kq8mZlZr9BxIiROKVF6J3AB4ljgHlKrzbtJK9FvBdxC6hp7IEcMZwFXSHoQuBc4krR46yUAkk4HdouIfQAiYqUJHSWNAZYVl9edX5U3MzPr8TprEZrUyb6tsk+xj5HWI/uvcgOIiKslbUBqcRoOPArsV7B42nBgi3LPVxeHnJEWU/3xgU5+zMzMepHOEqG9axVERFwEXNTBvgldHDsFmFLxoMo0eNZgL6ZqZmbWS3WcCAV31TCOXmvwvwZ7MVUzM7NeKs9bY10T2yLOrug5e7gFWy/wG2JmZma9VPcWXS0k1gQ+R5q0sH026ONW+7y9xILNF/gNMTMzs16q+4mQ2J6U/HyRtAq9gFmkeX0ai98QMzMz65Xyrj4/CPg88FXgP0jJD8DfgW8T3FnR6MzMzMyqqLwxQmInxIXAi8DPgQ8ADwHHZDX+5iTIzMzMepvOW4TEV0jdX6NJrT+zSV1fvyJ4LKtzfnVDNDMzM6uOrrrGJpMWW/0jcDlwM8HSqkdlZmZmVgPldI2JtL7X+0iLoZqZmZn1CV0lQnsAvwHeA5wOPIu4CfE5xMCqR2dmZmZWRZ0nQsF9BIeRFkH9JvA4aS2x3wEvodLLYpiZmZn1BuW9NRbMJbiA4P3AWNJ4oYGkleIBPob4NmKj6oRpZmZmVnn5l9gIHiD4MqmV6Ouk1+hHAGcCzyOuqWiEZmZmZlXS/bXGgvkEFxOMBnYlvVa/CDiwQrGZmZmZVVVlFl0NphNMBIYBX6vIOc3MzMyqbPUXXS0ULCTNPG1mZmbW41WmRcjMzMysF3IiZGZmZg3LiZCZmZk1LCdCZmZmDUzSnpKul/SCpJA0oWi/JE2S9KKkNkktkt5XVGdNSedLelXSwux8I4vqrC/pCklzs88Vktar/h12zomQmZlZYxsMPAp8C2grsf87wLeBY0jT5cwBpkoaUlDnHODTwMHAOGAd4EZJ/QrqXAnsAnyMtErFLsAVlbyR7qjsW2NmZmbWq0TETcBNAJKmFO6TJOBY4IyIuDYrO4yUDB0CXCppXeAI4PCImJrVORR4BvgwcKukbUnJzx4RcV9W52vAPZJGRcTMat9nR7rXIiSaECMQm5T8mJmZWU/RX1JrwWdijmM3I80ReFt7QUS0AXcDu2dFo4EBRXWeA2YU1BkLLADuKzj3vcDCgjp1ka9FSBwKfBfYtpNakfu8ZmZmVi1LImJMN48dlv2cXVQ+m7S8VnudpcCrJeoMK6jzSkRE+86ICElzCurURfkJi5gA/JJ0s/cAzwFLqhKVmZmZ9SRRtK0SZcWK65SqX855qipPy83/A94A9iCYUaV4zMzMrOd4Ofs5jNQA0m4oK1qJXgb6ARsCrxTVubugzlBJam8VysYfbcSqrU01lWeM0JbAH5wEmZmZNYynSEnM+PYCSWuR3gxrH+8zHVhcVGckaRhNe51ppLfTxhaceyzQxMrjhmouT4vQ68Db1QrEzMzMak/SYFJjB6QGkk0k7QS8HhHPSjoHOEnSP4F/ASeTBj5fCRARcyX9AvhpNubnNeAs4BHg9qzODEm3kN4y+yqpS+xS4MZ6vjEG+RKhG4FmhIj69ueZmZlZxYwB/lKwfWr2uRyYAJwJDAIuBNYHHgA+EhHzC445jjRu+Oqs7h3AlyJiaUGdLwDnseLtsuuBb1T4XnLLkwidSHrV7RLEtwkWVCkmMzMzq5GIaCG10HS0P4BJ2aejOm+TJlw8ppM6rwNf7GaYVZMnEboGeAv4CnAI4gngzRL1gmCfCsRmZmZmVlV5EqHmgl83ATt1UM/dZmZmZtYrlJ8IhdclMzMzs76lRyQ3ko6W9JSktyVNlzSuk7rbSfqLpNlZ/VmSTpM0sJYxm5mZWe9X96UwJB0EnAscDfw1+3mzpO0i4tkShywijWR/iDRG6f3AZaR7+U4tYjYzM7O+oXuJkBhJWmNkzZL7Y/lMkuU4HpgSEZdl28dI2hc4ivSm2sqnjngSeLKg6BlJzaTJnczMzMzKpoL1z8qozUeAs4FtOq0X9CvrdKk76y3g4Ii4pqD8QmD7iNirjHNsSZqL4PqI+F6J/ROBiQD9+/cfPXXq1HJCK9vDDz/MoEGDKnrO3qStra2h7x/8DMDPoNHvH/wMKnn/o0aNqsh52u29995vRURTRU/ah+RZdPU/SJMqvgJcQJor4C5gJqk1ZltSQvJQjutvSFqfpNSqth/uNBzpPmAXUqvUZcD3S9WLiMnAZICmpqZobm7OEV7XZs6cyZgx3V3Ut/drbW1t6PsHPwPwM2j0+wc/g0re/+jRoytyHitPnsHS3yctsbErwbeysr8QHAlsD/wXKXn5Qzfi6M6qtgeREqFDgP2A73bjumZmZtbA8iRCY4HrCV5c5fggCH4IzCBNy12uV4GlpFVtCxWualtSRDwXEY9HxO+A7wE/lFT3wd9mZmbWe+RJhNYFCt/iWkSaWLHQvcCe5Z4wIhaRVq0dX7RrPPlWo12D1M1X1tgkMzMzM8j31tgc0mJrhdtbFNUZQFpsLY+zgCskPUhKpI4ENgYuAZB0OrBbROyTbR9K6qL7BykZGwOcDvwhIt7JeW0zMzNrYHkSoX+xcuJzP/AxxNYE/0IMAz4NPJEngIi4WtIGwMnAcOBRYL+IeCarMrzouktIr9VvRRpL9AxpRdyz81zXzMzMLE8idAvwY8S7CF4nTYJ4IPAQ4nFSYjKEbkxqGBEXARd1sG9C0fbvgN/lvYaZmZlZsTxjhC4ljf9ZDEBwL/BZ4CnSW2MvAUcR/LrCMZqZmZlVRZ5FV+cBDxSV/Qn4U2VDMjMzM6uNHrHoqpmZmVk95J93R2xEGhS9LdBE8JWC8s2AfxC0VTBGMzMzs6rIlwiJI4DzgLVYMfvzV7K97wamkdb1+kXlQjQzMzOrjvK7xsR40ppd/wL+E7h4pf3Bo8BjwKcqFp2ZmZlZFeVpEfou6c2wvQjmIXYuUecR0lIcZmZmZj1ensHSY4Abs7fHOvI8q64bZmZmZtYj5UmEBgILu6izHmkRVTMzM7MeL08i9DQwuos6/wHM7HY0ZmZmZjWUJxG6DhiH+GzJveJwYEfg2grEZWZmZlZ1eQZLnwl8Hvgd4jPAugCIbwDjSOuOPQGcX+EYzczMzKoizxIbbyD2An4NK7UKnZf9vAc4hOhyHJGZmZlZj5BvQsXgWaAZsSPpNfkNgLnA/QTTKx+emZmZWfXkX2IDIHiENGeQmZmZWa/lRVfNzMysYXXeIiS+1K2zBr/u1nFmZmZmNdRV19gU0sKq5WpfiNWJkJmZmfV45YwRWgLcCDxe5VjMzMzMaqqrROguYE/SivJDgcuA3xO8XeW4zMzMzKqu88HSwd7AKOBnwJbAr4CXEOdnr9CbmZmZ9VpdvzUWPEnwXeA9wOeAB4CjgIcQDyKOQDRVN0wzMzOzyiv/9flgCcG1BPsCWwCnAcOBycCLiLHVCdHMzMysOro3j1DwDMEPgInAC8BgYKMKxmVmZmZWdflnlhYbA1/OPu8F3gZ+A/xfRSMzMzMzq7LyEiGxBvAJ4CvAvtlx/wC+BVxBMLdaAZqZmZlVS1czS28GHAEcThoPtBC4HLiM4MGqR2dmZmZWRV21CD2Z/WwFfgj8jmBhdUMyMzMzq42uEiEBi0mtQacAp6AuzxkE71390MzMzMyqq5wxQgOAkdUOxMzMzKzWOk+Eopuv15uZmZn1Ak50zMzMrGH1iERI0tGSnpL0tqTpksZ1UrdZ0nWSXpL0lqRHJH25lvGamZlZ31D3REjSQcC5pCU7dgbuA26WtEkHh+xOmsPoM8D2wMXAZEmH1CBcMzMz60PyzyxdeccDUyLismz7GEn7khZ2PbG4ckScVlR0saS9gU8DV1Y1UjMzM+tT6toiJGkgMBq4rWjXbaSWn3KtA7xRqbjMzMysMSgi6ndxaWPSoq17RcTdBeWnAF+IiFFlnOMTwJ+AD0bEKrNdS5pIWhyW/v37j546dWqlwgfg4YcfZtCgQRU9Z2/S1tbW0PcPfgbgZ9Do9w9+BpW8/1Gjuvzqy2Xvvfd+KyKaKnrSPqQndI0BFGdjKlG2CkkfJHWHfbNUEgQQEZOByQBNTU3R3Ny8epEWmTlzJmPGjKnoOXuT1tbWhr5/8DMAP4NGv3/wM6jk/Y8ePboi57Hy1Huw9KvAUmBYUflQYHZnB0raA7gZOCUiLq5OeGZmZtaX1TURiohFwHRgfNGu8aS3x0qStCcpCTo1Is6pWoBmZmbWp/WErrGzgCskPQjcCxwJbAxcAiDpdGC3iNgn224G/gxcBPxWUntr0tKIeKW2oZuZmVlvVvdEKCKulrQBcDJpcddHgf0i4pmsynBgi4JDJgBrA/8v+7R7Bti02vGamZlZ31H3RAggIi4itfCU2jehxPaEUnXNzMzM8qj3YGkzMzOzunEiZGZmZg3LiZCZmZk1LCdCZmZm1rCcCJmZmTUwSZMkRdHn5YL9yuq8KKlNUouk9xWdY01J50t6VdJCSddLGln7u8nPiZCZmZnNJE1X0/7ZoWDfd4BvA8cAuwJzgKmShhTUOQf4NHAwMI60GPqNkvpVPfLV1CNenzczM7O6WhIRLxcXShJwLHBGRFyblR1GSoYOAS6VtC5wBHB4REzN6hxKmt/vw8CtNbmDbnKLkJmZmW0u6QVJT0m6StLmWflmpPVAb2uvGBFtwN3A7lnRaGBAUZ3ngBkFdXosJ0JmZmZ9W39JrQWfiUX7HyBNVPwx4KukxOe+bNWH9mWsihdCn12wbxhpAfVXO6nTY7lrzMzMrG9bEhFjOtoZETcXbku6H5gFHAbc316t6DCVKCtWTp26c4uQmZmZLRcRC4DHgK2A9nFDxS07Q1nRSvQy0A/YsJM6PZYTITMzM1tO0lrANsBLwFOkRGd80f5xwH1Z0XRgcVGdkcC2BXV6LHeNmZmZNTBJPwNuAJ4lteL8AGgCLo+IkHQOcJKkfwL/Ak4GFgBXAkTEXEm/AH4qaQ7wGnAW8Ahwe41vJzcnQmZmZo1tJPA7UtfWK6RxQR+IiGey/WcCg4ALgfVJg6s/EhHzC85xHLAEuDqrewfwpYhYWpM7WA1OhMzMzBpYRHy+i/0BTMo+HdV5mzTh4jGVjK0WPEbIzMzMGpYTITMzM2tYToTMzMysYTkRMjMzs4blRMjMzMwalhMhMzMza1hOhMzMzKxhOREyMzOzhuVEyMzMzBqWEyEzMzNrWE6EzMzMrGE5ETIzM7OG5UTIzMzMGpYTITMzM2tYToTMzMysYTkRMjMzs4blRMjMzMwalhMhMzMza1g9IhGSdLSkpyS9LWm6pHGd1F1L0hRJj0haLKmlhqGamZlZH1L3REjSQcC5wGnAzsB9wM2SNungkH7A28AFwJ9rEqSZmZn1SXVPhIDjgSkRcVlEzIiIY4CXgKNKVY6IhRFxZERMBp6vZaBmZmbWtygi6ndxaSDwFnBwRFxTUH4hsH1E7NXF8Rdk9Zo7qTMRmAjQv3//0VOnTq1E6Ms9/PDDDBo0qKLn7E3a2toa+v7BzwD8DBr9/sHPoJL3P2rUqIqcp93ee+/9VkQ0VfSkfUj/Ol9/Q1JX1+yi8tnAhytxgazlaDJAU1NTNDc3V+K0y82cOZMxY8ZU9Jy9SWtra0PfP/gZgJ9Bo98/+BlU8v5Hjx5dkfNYeXpC1xhAcbOUSpSZmZmZVVS9E6FXgaXAsKLyoazaSmRmZmZWUXVNhCJiETAdGF+0azzp7TEzMzOzqqn3GCGAs4ArJD0I3AscCWwMXAIg6XRgt4jYp/0ASdsBA0ljjAZL2gkgIh6uaeRmZmbWq9U9EYqIqyVtAJwMDAceBfaLiGeyKsOBLYoOuwl4b8H2Q9lPVTNWMzMz61vqnggBRMRFwEUd7JtQomzTKodkZmZmDaDeg6XNzMzM6saJkJmZmTUsJ0JmZmbWsJwImZmZWcNyImRmZmYNy4mQmZmZNSwnQmZmZtawnAiZmZlZw3IiZGZmZg3LiZCZmZk1LCdCZmZm1rCcCJmZmVnDciJkZmZmDcuJkJmZmTUsJ0JmZmbWsJwImZmZWcNyImRmZmYNy4mQmZmZNSwnQmZmZtawnAiZmZlZw3IiZGZmZg3LiZCZmZk1LCdCZmZm1rCcCJmZmVnDciJkZmZmDcuJkJmZmTUsJ0JmZmbWsJwImZmZWcNyImRmZmYNy4mQmZmZNSwnQmZmZtawnAiZmZlZw+oRiZCkoyU9JeltSdMljeui/g6S7pLUJukFSadIUq3iNTMz62vyfhf3FXVPhCQdBJwLnAbsDNwH3Cxpkw7qrwNMBWYDuwLfBE4Ajq9JwGZmZn1M3u/ivqTuiRApgZkSEZdFxIyIOAZ4CTiqg/pfANYGDouIRyPiWuC/gePdKmRmZtYteb+L+4y6JkKSBgKjgduKdt0G7N7BYWOBeyKiraDsVmBjYNNKx2hmZtaXdfO7uM/oX+frbwj0I3VzFZoNfLiDY4YBz5eo377vqcIdkiYCE7PNkFSYQFVCf2BJhc/ZmzT6/YOfAfgZNPr9g59BT77/QZJaC7YnR8Tkgu3ufBf3GfVOhNpF0bZKlHVVv1Q52W/25OLySpHUGhFjqnX+nq7R7x/8DMDPoNHvH/wM+sj95/0u7hPqPUboVWApqSWn0FBWzUzbvdxBfTo5xszMzErrzndxn1HXRCgiFgHTgfFFu8aTRqyXMg0YJ2mtovovAk9XOkYzM7O+rJvfxX1GvVuEAM4CJkj6iqRtJZ1LGvh8CYCk0yXdUVD/SuAtYIqk7SUdCHwPOCsi6tGEV7Vut16i0e8f/AzAz6DR7x/8DHr7/Xf6XdyXqT65Q1EQ0tHAd4DhwKPAcRFxd7ZvCtAcEZsW1N8BuBDYDXiD9Bv1ozolQmZmZr1eZ9/FfVmPSITMzMzM6qEndI2ZmZmZ1YUTITMzM2tYToS6qVEXpwOQtKek67MFb0PShHrHVEuSTpT0N0nzJL0i6QZJ29c7rlqS9HVJj2TPYJ6kaZI+Xu+46kXS97O/CxfUO5ZakTQpu+fCz8v1jqvWJA2XdHn2b8Hbkh6XtFe947LyORHqhkZenC4zmDSQ7ltApWfq7g2agYtIU89/iDSb7O2S3lXPoGrseeC7wC7AGOBO4H8l7VjXqOpA0geArwKP1DuWOphJGljb/tmhvuHUlqT1gHtJEw9+HNgWOAaYU8ewLCcPlu4GSQ8Aj0TEVwvKngD+EBEn1i+y2pO0APhGREypdyz1ImkwMBf4VETcUO946kXS68CJEXFpvWOpFUnrAv9HSoROAR6NiG/UN6rakDQJ+ExENFRraCFJpwF7RcQH6x2LdZ9bhHJq9MXprKQhpL9Lb9Q7kHqQ1E/S50kthX1+8rUik0n/Abqz3oHUyeZZF/lTkq6StHm9A6qxTwEPSLpa0hxJD0v6hiR1daD1HE6E8utscbri6cmtMZwLPEya9bxhSNohaxF8hzSX139GxD/qHFbNSPoqsCXwg3rHUicPABOAj5FaxIYB90naoJ5B1djmwNHALOCjpH8LzgC+Xs+gLJ+esuhqb9SQi9PZyiSdBewB7BERS+sdT43NBHYC1gM+DVwuqTkiHq1nULUgaRRpjOC4bHmChhMRNxduS7qflBAcRpqluBGsAbQWDIl4SNJWpESoYQbO93ZuEcqvoRensxUknQ0cDHwoImbVO55ai4hFEfFkRLR/ETwMHFfnsGplLKl1+FFJSyQtAfYCjs6216xveLUXEQuAx4Ct6h1LDb0EPF5UNgNolBdn+gQnQjk1+uJ0lmTr8BxCSoL+We94eog1gEZJAP6X9IbUTgWfVuCq7NcN10qULYS9DSk5aBT3AqOKyrYGnqlDLNZN7hrrnrOAKyQ9SPqLcCQNsjgdLH9Lastscw1gE0k7Aa9HxLN1C6xGJF0IHEoaKPmGpPbWwQXZ/4r7PElnAH8GniMNFj+ENK1AQ8wlFBFvAm8WlklaSPo70Oe7BgEk/Qy4AXiW1CL+A6AJuLyecdXY2aRxUScBV5OmU/km8P26RmW5+PX5bmrUxekAJDUDfymx6/KImFDTYOpAUkd/aU6NiEm1jKVessWQ9yZ1Ec8lzaHz04i4tZ5x1ZOkFhrr9fmrgD1JXYSvAPcDP4iI4q6iPi2bSPQ0UsvQs6SxQed7EfDew4mQmZmZNSyPETIzM7OG5UTIzMzMGpYTITMzM2tYToTMzMysYTkRMjMzs4blRMjMzMwalhMhsxqQ1CwpJE2qdyx5SJqUxd1cwXO1f+o6AamkKVkcmxaUbZqVTalfZJ2TtGHRc/QcKGarwYmQWQOTNCH7Mp1Qw8teDpwK3FjDa/Ylb5Ge36l4KQez1eYlNsxq40FgW9KivY1uSkS01DuIDrxA+n2aW+9AOhIRbwGTYPks7++tYzhmvZ4TIbMayL68vDhrDxcRi/Hvk1lDcdeYWYGsq+haSbMktUmaJ+leSV8sUbd4zMsqn4K6JccISWrJygdIOkXSvyW9Lemfkr5aUO9ISf/IYnpe0qmS1ig6V6fjkCQ9LenpwmsDv8o2f1UU+6Yljv+MpAclvSXpdUlXSRpRxmMtS8GYnc0lHSPpkex+W7L9AyV9Q9JNkp6R9E4Wx+2SPtbJeT8s6R5JC7P6/ytpmw7qlhwjJGlrSWdIapX0SnbtZyRNljSyxHmW/15I2knSnyW9mT27uyTtXuKYIZJ+IOnR7M/d/OzPw9WSRud9nmZWHrcIma3sYuBx4G7gJWADYD/gCkmjIuIHBXVbOjjHe4AvA205rnsV8B/ATcBi4DPAZEmLgR2Bw0hjau4A9gdOIY0V+e8c1yg2hbSC+gHAdcDDBfveLKp7dHbd64G7slgPAt4vaaeIeGc14ih2LjCOtLr9TcDSrPxd2b77gKmkhT6HA58EbpL01Yj4eeGJJH2GtCr4ouznS8AewDTSQrHlOhA4krTY8H3Z+d4HfAX4pKQxEfFCiePGkBZnngb8HNgE+DRwR/bcZmZxCrgF2L2g7hLSn6Vm4B5geo54zaxcEeGPP/5kH2CLEmUDSQnIYmBEF8evQ/qCXQocWFDeDAQwqah+S1b+N2C9gvLNSV+2bwBPFV4XWI801ugVoH9X1yjY/zTwdFHZhOyYCR0cMynbPw/YoWjfldm+z5X5bNvP1dzB/inZ/heAzUrsXxMYWaJ8XeBR4HVgUEH5YOC17PdtTNExZ2fXCmDTgvJNs7IpRfVHAGuWuPZHst/ri4vKmwvOP6Fo39ey8osKynbIyv5U4hprAOt38Mxa0j/j9f+7448/vfXjrjGzAhHx7xJli4ALSS2o+3R0rKT+wDWkL7UTIuKPOS79vYh4s+Cas4C/kpKe/4qC1oas3g3AhqQv6Fo4LyL+UVR2WfZztwpf68yIeKq4MCLeiYjnS5TPBX4JrA/sWrDrAFIr0pUR0Vp02CRyDIiOiBeiRKtXRNwGPAZ8tIND742IKUVlvyS19pR6bqu0IkbEsoh4o9xYzSwfd42ZFZC0CfBdUsKzCTCoqEpnicfFpBaCiyLirJyXLv6iBngx+1mqS6Q9MRpJbV6hLhXfc9nP9St8rQc72iHpfcAJwJ6kbrG1iqoU/v7skv28q/g8ETFX0sPAXuUElHVdfYHUgvZ+0j33K6iyqINDV3luEbFY0mxWfm6Pk7omD5b0XlJX5V+B1iwRN7MqcSJklpG0OelLeH3SmIzbSK0GS0ldJoeRumdKHXsiabzIn4Fv5r121qpRbEn2s7N9A/Jeq5ve7CSGfiX2rY6XSxVK+gBwJ+nfrTtI45XmAcuAnUgtQIW/P+tmP2fnuU4HzgKOJY0xupWUiLa33kyg41fY3+ygfAkFzy0ilkr6EGns12dYMfZrvqTLgRMjYkGOeM2sTE6EzFY4njQ4+vDi7gxJB5MSoVVIOgj4CfAQ8PmIWFqqXg0sy3529Pd6XXrw/DgFOpop+WRSC93eUTQPUZaIHlBUv/1e393B+YaVE4ykoaTk9lFg94iYX7T/4HLO05Ws++s44DhJW5Jaq74GfIPURXpoJa5jZivzGCGzFbbMfl5bYl/JLpTsNegppBaCT9T5f+3t40jeU7wj+2Jdr8Qx7UlbpVt1qmFL4PXiJChT6vfn/zraJ2ldUitSOTYn/Vt5W4kkaGS2v6Ii4smI+AUp9gWsmuSZWYU4ETJb4ensZ3NhoaSPkrq9KCrfkjSWYzHw8Yh4sbhOjf2T1FV0QNaKAYCkQcB5HRzzWvZzkyrHVglPA++StGNhoaQjKD1Y+TpScniIpDFF+yaxouusnOsC7CFpecIoaTBpwPhqt6xL2iwb/1RsfVJ3X56pGMwsB3eNma1wEXA4cI2ka0mtPNsD+wK/J82bU+g80ptbdwIHSjqw+IQRMamaARdda7Gkc4EfAA9J+hPp7/h40sDrUonaNNJ8RMdKehcrxtOc38G4pXo6h5Tw/FXS70ldX2NI8wL9gTS2ZrmIWCBpImn+oHskFc4jtD1prqg9u7poRLws6Srg88DDkm4jJVHjgbdJg5x3Ws17ez/wJ0nTSV1wLwIbkVqCBrB680WZWSecCJllIuIRSXsDPyZNotgf+DtpMr03WTURWjv7+aHsU8qkigfauR+SEpuvAhNJA4KvyuJ4vLhyRLwh6dPZcYcDTdmu39DDxhNFxC2SPkkaK3QQqVvvQWBvUvfUZ0oc8wdJ+5Lu73PAO6QEaCzwPcpIhDJHALOy636dNIfT9aTBzaW6UvNqBU4ndYXtS2oJeoX0xuB5EXFzBa5hZiUooqNxiWZmlZMt/fFDSgx2tu7Jlh/ZKyJU71jMeiuPETKzWvtLtg7XJfUOpDeStKFWrGVX1jxIZtYxd42ZWa20FG2XmqTRuvYWcGq9gzDrK9w1ZmZmZg3LXWNmZmbWsJwImZmZWcNyImRmZmYNy4mQmZmZNSwnQmZmZtawnAiZmZlZw/r/E5C15QCWZAkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 576x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "counts, edges = np.histogram(azimuth_non_scaled['azimuth'],bins=80)\n",
    "accuracies = []\n",
    "errors = []\n",
    "nrn = []\n",
    "for a, b in zip(edges[:-1], edges[1:]):\n",
    "    fltr = (a < azimuth_non_scaled['azimuth']) & (azimuth_non_scaled['azimuth'] < b)\n",
    "    accuracies.append(np.mean(pr[fltr]))\n",
    "    errors.append(np.std(pr[fltr]) / np.sqrt(len(pr[fltr])))\n",
    "    #nrn.append(len(pr[fltr][pr[fltr]>0.5])/len(pr[fltr]))\n",
    "\n",
    "\n",
    "fig = plt.figure(figsize=(8,8))\n",
    "\n",
    "ax = fig.add_subplot(111)\n",
    "ax.errorbar((edges[:-1] + edges[1:])/2, accuracies, fmt='.', color='magenta', label='azimuth', yerr = errors, ecolor='red')\n",
    "#ax.plot((edges[:-1] + edges[1:])/2, nrn, label='Ratio of correctness')\n",
    "\n",
    "#ax.plot((edges[:-1] + edges[1:])/2,stest)\n",
    "ax.grid()\n",
    "ax.set_ylim(0,1)\n",
    "ax.set_xlabel('azimuth [radians]', fontsize=20)\n",
    "ax.legend(fontsize=16)\n",
    "ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\n",
    "ax.tick_params(axis='x',labelsize=14)\n",
    "ax.tick_params(axis='y',labelsize=14)\n",
    "ax.set_ylabel('Mean Absolute Residuals',fontsize=20,color='magenta')\n",
    "\n",
    "ax2 = ax.twinx()\n",
    "ax2.hist(azimuth_non_scaled['azimuth'],bins=80,alpha=0.2,color='k');\n",
    "ax2.set_ylabel('Counts', fontsize=20)\n",
    "ax2.tick_params(axis='y',labelsize=14)\n",
    "\n",
    "print(min(accuracies),' lavest mean residual')\n",
    "print(np.argmin(accuracies))\n",
    "qpr = (edges[:-1] + edges[1:])/2\n",
    "print(qpr[35],' bin center')\n",
    "print((abs(min(azimuth_non_scaled['azimuth'])-max(azimuth_non_scaled['azimuth'])))/100,' afvigelse')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2368, 2113, 2064, 2246, 2378, 2501, 2590, 2557, 2630, 2539, 2664,\n",
       "       2569, 2509, 2456, 2146, 2047, 2211, 2480, 2620, 2595, 2710, 2728,\n",
       "       2712, 2769, 2681, 2557, 2417, 2042, 1837, 2033, 2383, 2667, 2663,\n",
       "       2746, 2702, 2678, 2667, 2687, 2557, 2461, 2279, 2028, 1967, 2177,\n",
       "       2418, 2500, 2502, 2635, 2595, 2521, 2595, 2540, 2452, 2395, 2126,\n",
       "       2038, 2318, 2570, 2669, 2711, 2842, 2828, 2857, 2868, 2816, 2734,\n",
       "       2594, 2289, 2000, 2105, 2481, 2681, 2694, 2812, 2788, 2800, 2818,\n",
       "       2767, 2697, 2513], dtype=int64)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "34        6.331242\n",
       "249       5.899983\n",
       "408       6.401597\n",
       "480       6.786203\n",
       "488       5.754976\n",
       "            ...   \n",
       "199556    6.673407\n",
       "199692    6.531184\n",
       "199815    6.099558\n",
       "199897    6.265499\n",
       "199950    6.156023\n",
       "Name: azimuth, Length: 2513, dtype: float64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "A=edges[-2]\n",
    "B=edges[-1]\n",
    "fltr = (A <= azimuth_non_scaled['azimuth']) & (azimuth_non_scaled['azimuth'] <= B)\n",
    "accuracies.append(np.mean(pr[fltr]))\n",
    "errors.append(np.std(pr[fltr]) / np.sqrt(len(pr[fltr])))\n",
    "errors\n",
    "pr[fltr]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.44660104436744513"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.std(pr[fltr])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "50.12983143797713"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sqrt(len(pr[fltr]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0088"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "0.44/50"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Med retro"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "dbfile = 'dev_lvl7_mu_nu_e_classification_v003.db'\n",
    "with sqlite3.connect(dbfile) as con:\n",
    "    query = 'select * from truth WHERE pid == 13'\n",
    "    t = pd.read_sql(query,con)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fejl herunder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#pred = np.array(pred)\n",
    "#truth = np.array(truth)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "list indices must be integers or slices, not tuple",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-43-467ab75aebe8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mfig\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0max\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msubplots\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfigsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m16\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0max\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marccos\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mazimuth_pred\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mazimuth\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtruth\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Azimuth'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0.1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0max\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'k--'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0max\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'equal'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0max\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_xlabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Prediction'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: list indices must be integers or slices, not tuple"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA6oAAAEzCAYAAAA4iv68AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAUR0lEQVR4nO3dX4il93kf8O/TVQSJk0Ym2gR3JVG1KJG3xSr2RDWhf5SaNlr1Ygn4QnKoiQgsAivkUqIXScA3zUUhGMteFiOEb6KbmFQpikVpSVxw1GgFtqy1kZnKxJoooFUcUrChYu2nFzNNJpNZzTu755z5nfd8PjAw7zk/zTwPZ/mi75xzZqq7AwAAAKP4eyc9AAAAAOynqAIAADAURRUAAIChKKoAAAAMRVEFAABgKIoqAAAAQzmyqFbV01X1VlW9ep37q6o+VVXbVfVKVX1w8WMCLJesAzaFvAPWwZRnVJ9J8uC73H8uyT17HxeSfPbmxwJYuWci64DN8EzkHTC4I4tqd38pyXfe5cj5JJ/vXS8mua2q3reoAQFWQdYBm0LeAetgEe9RPZPkjX3XO3u3AcyJrAM2hbwDTtwtC/gadchtfejBqgvZfQlJ3vOe93zo3nvvXcC3B+bk5Zdffru7T5/0HIeQdcDCDJx1ycS8k3XAUW4m6xZRVHeS3Lnv+o4kbx52sLsvJbmUJFtbW3358uUFfHtgTqrqT096huuQdcDCDJx1ycS8k3XAUW4m6xbx0t/nknx87zfEfTjJX3X3ny/g6wKMRNYBm0LeASfuyGdUq+p3kjyQ5Paq2knyG0l+KEm6+2KS55M8lGQ7yfeSPLqsYQGWRdYBm0LeAevgyKLa3Y8ccX8n+cTCJgI4AbIO2BTyDlgHi3jpLwAAACyMogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADGVSUa2qB6vqtararqonD7n/x6vq96vqq1V1paoeXfyoAMsl64BNIOuAdXBkUa2qU0meSnIuydkkj1TV2QPHPpHk6919X5IHkvznqrp1wbMCLI2sAzaBrAPWxZRnVO9Pst3dr3f3O0meTXL+wJlO8mNVVUl+NMl3klxb6KQAyyXrgE0g64C1MKWonknyxr7rnb3b9vt0kvcneTPJ15L8Wnf/4OAXqqoLVXW5qi5fvXr1BkcGWApZB2wCWQeshSlFtQ65rQ9c/0KSryT5B0n+WZJPV9Xf/zv/Ufel7t7q7q3Tp08fc1SApZJ1wCaQdcBamFJUd5Lcue/6juz+hG2/R5N8oXdtJ/lWknsXMyLASsg6YBPIOmAtTCmqLyW5p6ru3nsj/cNJnjtw5ttJPpIkVfVTSX4myeuLHBRgyWQdsAlkHbAWbjnqQHdfq6rHk7yQ5FSSp7v7SlU9tnf/xSSfTPJMVX0tuy8peaK7317i3AALJeuATSDrgHVxZFFNku5+PsnzB267uO/zN5P8u8WOBrBasg7YBLIOWAdTXvoLAAAAK6OoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAok4pqVT1YVa9V1XZVPXmdMw9U1Veq6kpV/dFixwRYPlkHbAJZB6yDW446UFWnkjyV5N8m2UnyUlU9191f33fmtiSfSfJgd3+7qn5ySfMCLIWsAzaBrAPWxZRnVO9Pst3dr3f3O0meTXL+wJmPJflCd387Sbr7rcWOCbB0sg7YBLIOWAtTiuqZJG/su97Zu22/n07y3qr6w6p6uao+vqgBAVZE1gGbQNYBa+HIl/4mqUNu60O+zoeSfCTJDyf546p6sbu/+be+UNWFJBeS5K677jr+tADLI+uATSDrgLUw5RnVnSR37ru+I8mbh5z5Ynd/t7vfTvKlJPcd/ELdfam7t7p76/Tp0zc6M8AyyDpgE8g6YC1MKaovJbmnqu6uqluTPJzkuQNn/kuSf1lVt1TVjyT550m+sdhRAZZK1gGbQNYBa+HIl/5297WqejzJC0lOJXm6u69U1WN791/s7m9U1ReTvJLkB0k+192vLnNwgEWSdcAmkHXAuqjug29LWI2tra2+fPnyiXxvYFxV9XJ3b530HIsi64DDyDpgE9xM1k156S8AAACsjKIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKFMKqpV9WBVvVZV21X15Luc+9mq+n5VfXRxIwKshqwDNoGsA9bBkUW1qk4leSrJuSRnkzxSVWevc+63kryw6CEBlk3WAZtA1gHrYsozqvcn2e7u17v7nSTPJjl/yLlfTfK7Sd5a4HwAqyLrgE0g64C1MKWonknyxr7rnb3b/lpVnUnyi0kuLm40gJWSdcAmkHXAWphSVOuQ2/rA9W8neaK7v/+uX6jqQlVdrqrLV69enTgiwErIOmATyDpgLdwy4cxOkjv3Xd+R5M0DZ7aSPFtVSXJ7koeq6lp3/97+Q919KcmlJNna2joYigAnSdYBm0DWAWthSlF9Kck9VXV3kj9L8nCSj+0/0N13///Pq+qZJP/1YJgBDE7WAZtA1gFr4cii2t3Xqurx7P7Wt1NJnu7uK1X12N793r8ArD1ZB2wCWQesiynPqKa7n0/y/IHbDg2y7v7lmx8LYPVkHbAJZB2wDqb8MiUAAABYGUUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABiKogoAAMBQFFUAAACGoqgCAAAwFEUVAACAoSiqAAAADEVRBQAAYCiKKgAAAENRVAEAABjKpKJaVQ9W1WtVtV1VTx5y/y9V1St7H1+uqvsWPyrAcsk6YBPIOmAdHFlUq+pUkqeSnEtyNskjVXX2wLFvJfnX3f2BJJ9McmnRgwIsk6wDNoGsA9bFlGdU70+y3d2vd/c7SZ5Ncn7/ge7+cnf/5d7li0nuWOyYAEsn64BNIOuAtTClqJ5J8sa+6529267nV5L8wWF3VNWFqrpcVZevXr06fUqA5ZN1wCaQdcBamFJU65Db+tCDVT+f3UB74rD7u/tSd29199bp06enTwmwfLIO2ASyDlgLt0w4s5Pkzn3XdyR58+ChqvpAks8lOdfdf7GY8QBWRtYBm0DWAWthyjOqLyW5p6rurqpbkzyc5Ln9B6rqriRfSPIfuvubix8TYOlkHbAJZB2wFo58RrW7r1XV40leSHIqydPdfaWqHtu7/2KSX0/yE0k+U1VJcq27t5Y3NsBiyTpgE8g6YF1U96FvS1i6ra2tvnz58ol8b2BcVfXynP6HSNYBh5F1wCa4mayb8tJfAAAAWBlFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDUVQBAAAYiqIKAADAUBRVAAAAhqKoAgAAMBRFFQAAgKEoqgAAAAxFUQUAAGAoiioAAABDmVRUq+rBqnqtqrar6slD7q+q+tTe/a9U1QcXPyrAcsk6YBPIOmAdHFlUq+pUkqeSnEtyNskjVXX2wLFzSe7Z+7iQ5LMLnhNgqWQdsAlkHbAupjyjen+S7e5+vbvfSfJskvMHzpxP8vne9WKS26rqfQueFWCZZB2wCWQdsBamFNUzSd7Yd72zd9txzwCMTNYBm0DWAWvhlgln6pDb+gbOpKouZPclJEnyf6vq1Qnff53cnuTtkx5igea2TzK/nea2T5L8zAl9X1k33dz+3c1tn2R+O81tn0TWrYO5/bub2z7J/Haa2z7JTWTdlKK6k+TOfdd3JHnzBs6kuy8luZQkVXW5u7eONe3g5rbT3PZJ5rfT3PZJdnc6oW8t6yaa205z2yeZ305z2yeRdetgbjvNbZ9kfjvNbZ/k5rJuykt/X0pyT1XdXVW3Jnk4yXMHzjyX5ON7vyXuw0n+qrv//EaHAjgBsg7YBLIOWAtHPqPa3deq6vEkLyQ5leTp7r5SVY/t3X8xyfNJHkqyneR7SR5d3sgAiyfrgE0g64B1MeWlv+nu57MbWvtvu7jv807yiWN+70vHPL8O5rbT3PZJ5rfT3PZJTnAnWTfZ3Haa2z7J/Haa2z6JrFsHc9tpbvsk89tpbvskN7FT7WYRAAAAjGHKe1QBAABgZZZeVKvqwap6raq2q+rJQ+6vqvrU3v2vVNUHlz3TzZiwzy/t7fFKVX25qu47iTmP46id9p372ar6flV9dJXzHdeUfarqgar6SlVdqao/WvWMxzXh392PV9XvV9VX93Ya+v1EVfV0Vb11vT9lsG65kMg6Wbd6sk7WnYS5ZV0yv7ybW9Yl88s7WTcxF7p7aR/ZfZP+/07yj5LcmuSrSc4eOPNQkj/I7t/s+nCS/7XMmVawz88lee/e5+dG3mfqTvvO/Y/svqfloyc9900+Rrcl+XqSu/auf/Kk517ATv8xyW/tfX46yXeS3HrSs7/LTv8qyQeTvHqd+9cmF47xGK3NTrJO1g28k6wb/zGa405rk3dzy7pjPEZrk3eybnouLPsZ1fuTbHf36939TpJnk5w/cOZ8ks/3rheT3FZV71vyXDfqyH26+8vd/Zd7ly9m92+PjWzKY5Qkv5rkd5O8tcrhbsCUfT6W5Avd/e0k6e457NRJfqyqKsmPZjfQrq12zOm6+0vZnfF61ikXElkn61ZP1sm6kzC3rEvml3dzy7pkfnkn6ybmwrKL6pkkb+y73tm77bhnRnHcWX8luz89GNmRO1XVmSS/mORixjflMfrpJO+tqj+sqper6uMrm+7GTNnp00nen90/yP61JL/W3T9YzXhLsU65kMg6Wbd6sk7WnYS5ZV0yv7ybW9Yl88s7WTcxFyb9eZqbUIfcdvDXDE85M4rJs1bVz2c3zP7FUie6eVN2+u0kT3T393d/sDO0KfvckuRDST6S5IeT/HFVvdjd31z2cDdoyk6/kOQrSf5Nkn+c5L9V1f/s7v+z5NmWZZ1yIZF1sm71ZJ2sOwlzy7pkfnk3t6xL5pd3sm7Xkbmw7KK6k+TOfdd3ZPcnA8c9M4pJs1bVB5J8Lsm57v6LFc12o6bstJXk2b0wuz3JQ1V1rbt/byUTHs/Uf3Nvd/d3k3y3qr6U5L4kI4ZZMm2nR5P8p959I8B2VX0ryb1J/mQ1Iy7cOuVCIutk3erJOll3EuaWdcn88m5uWZfML+9k3dRcmPJG1hv9yG4Rfj3J3fmbNwv/kwNn/n3+9ptr/2SZM61gn7uSbCf5uZOed1E7HTj/TAZ+0/3Ex+j9Sf773tkfSfJqkn960rPf5E6fTfKbe5//VJI/S3L7Sc9+xF7/MNd/0/3a5MIxHqO12UnWybqBd5J14z9Gc9xpbfJubll3jMdobfJO1k3PhaU+o9rd16rq8SQvZPc3XD3d3Veq6rG9+y9m97eNPZTdAPhedn+CMKSJ+/x6kp9I8pm9n1Rd6+6tk5r5KBN3WhtT9unub1TVF5O8kuQHST7X3Yf+Ou0RTHyMPpnkmar6WnZD4InufvvEhj5CVf1OkgeS3F5VO0l+I8kPJeuXC4msk3WrJ+tk3UmYW9Yl88u7uWVdMr+8k3XTc6H2Wi4AAAAMYdm/9RcAAACORVEFAABgKIoqAAAAQ1FUAQAAGIqiCgAAwFAUVQAAAIaiqAIAADAURRUAAICh/D/EvZ4d2wr5ZgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1152x360 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(1,3, figsize=(16,5))\n",
    "ax[0].scatter(np.arccos(azimuth_pred[:,1]), np.azimuth(truth[:,1]), label='Azimuth', s=0.1)\n",
    "ax[0].plot([0, 1], [0, 1], 'k--')\n",
    "ax[0].axis('equal')\n",
    "ax[0].set_xlabel('Prediction')\n",
    "ax[0].set_ylabel('Target')\n",
    "ax[0].set_title(f'Azimuth | Correlation {np.round(np.corrcoef(np.arccos(azimuth_pred[:,1]), np.arccos(azimuth[:,1]))[0,1],3)}')\n",
    "\n",
    "ax[1].scatter((azimuth_pred[:,0]), (azimuth[:,0]), label='Azimuth', s=0.1)\n",
    "ax[1].plot([-1, 1], [-1, 1], 'k--')\n",
    "ax[1].axis('equal')\n",
    "#ax[1].text(0.0, 1.0, 'Correlation: '+ str())\n",
    "ax[1].set_xlabel('Prediction')\n",
    "ax[1].set_ylabel('Target')\n",
    "ax[1].set_title(f'sin(Azimuth) | Correlation {np.round(np.corrcoef(azimuth_pred[:,0], azimuth[:,0])[0,1],3)}')\n",
    "\n",
    "ax[2].scatter((azimuth_pred[:,1]), (azimuth[:,1]), label='Azimuth', s=0.1)\n",
    "ax[2].plot([0, 1], [0, 1], 'k--')\n",
    "ax[2].axis('equal')\n",
    "#ax[2].text(0.0, 0.9, 'Correlation: '+ str()\n",
    "ax[2].set_xlabel('Prediction')\n",
    "ax[2].set_ylabel('Target')\n",
    "ax[2].set_title(f'cos(Azimuth) | Correlation {np.round(np.corrcoef(azimuth_pred[:,1], azimuth[:,1])[0,1],3)}')\n",
    "fig.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1,3, figsize=(16,5))\n",
    "ax[0].scatter(np.arccos(pred[:,1]), np.arccos(truth[:,1]), label='Azimuth', s=0.1)\n",
    "ax[0].plot([0, 1], [0, 1], 'k--')\n",
    "ax[0].axis('equal')\n",
    "ax[0].set_xlabel('Prediction')\n",
    "ax[0].set_ylabel('Target')\n",
    "ax[0].set_title(f'Azimuth | Correlation {np.round(np.corrcoef(np.arccos(pred[:,1]), np.arccos(truth[:,1]))[0,1],3)}')\n",
    "\n",
    "ax[1].scatter((pred[:,0]), (truth[:,0]), label='Azimuth', s=0.1)\n",
    "ax[1].plot([-1, 1], [-1, 1], 'k--')\n",
    "ax[1].axis('equal')\n",
    "#ax[1].text(0.0, 1.0, 'Correlation: '+ str())\n",
    "ax[1].set_xlabel('Prediction')\n",
    "ax[1].set_ylabel('Target')\n",
    "ax[1].set_title(f'sin(Azimuth) | Correlation {np.round(np.corrcoef(pred[:,0], truth[:,0])[0,1],3)}')\n",
    "\n",
    "ax[2].scatter((pred[:,1]), (truth[:,1]), label='Azimuth', s=0.1)\n",
    "ax[2].plot([0, 1], [0, 1], 'k--')\n",
    "ax[2].axis('equal')\n",
    "#ax[2].text(0.0, 0.9, 'Correlation: '+ str()\n",
    "ax[2].set_xlabel('Prediction')\n",
    "ax[2].set_ylabel('Target')\n",
    "ax[2].set_title(f'cos(Azimuth) | Correlation {np.round(np.corrcoef(pred[:,1], truth[:,1])[0,1],3)}')\n",
    "fig.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1,3, figsize=(16,5))\n",
    "ax[0].hist2d((pred[:,0]), (truth[:,0]), label='Azimuth')\n",
    "ax[0].plot([-1, 1], [-1, 1], 'k--')\n",
    "ax[0].axis('equal')\n",
    "\n",
    "ax[0].set_xlabel('Prediction')\n",
    "ax[0].set_ylabel('Target')\n",
    "ax[0].set_title(f'Azimuth | Correlation: {np.round(np.corrcoef(pred[:,0], truth[:,0])[0,1],3)}')\n",
    "\n",
    "\n",
    "ax[1].hist2d(np.arccos(pred[:,1]), np.arccos(truth[:,1]), label='Azimuth')\n",
    "ax[1].plot([0, 1], [0, 1], 'k--')\n",
    "ax[1].axis('equal')\n",
    "ax[1].set_xlabel('Prediction')\n",
    "ax[1].set_ylabel('Target')\n",
    "ax[1].set_title(f'sin(Azimuth) | Correlation: {np.round(np.corrcoef(np.arccos(pred[:,1]), np.arccos(truth[:,1]))[0,1],3)}')\n",
    "\n",
    "ax[2].hist2d((pred[:,1]), (truth[:,1]), label='Azimuth')\n",
    "ax[2].plot([0, 1], [0, 1], 'k--')\n",
    "ax[2].axis('equal')\n",
    "ax[2].set_xlabel('Prediction')\n",
    "ax[2].set_ylabel('Target')\n",
    "ax[2].set_title(f'cos(Azimuth) | Correlation: {np.round(np.corrcoef(pred[:,1], truth[:,1])[0,1],3)}')\n",
    "fig.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plt.hist(truth[:,1], histtype='step', bins=50)\n",
    "#plt.hist(pred[:,1] , histtype='step', bins=50)\n",
    "plt.hist(np.arccos(truth[:,1]))\n",
    "plt.hist(np.arccos(pred[:,1]));\n",
    "#plt.plot(np.cos(np.linspace(0,1.6)));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#with open('ES_predictions','wb') as f:\n",
    "#    pkl.dump(predictions, f)\n",
    "#with open('ES_truths','wb') as f:\n",
    "#    pkl.dump(truth, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_bins = 100\n",
    "fig, ax = plt.subplots(3,1, figsize=(10,7))\n",
    "ax[0].hist(pred[:,0] , bins=n_bins, histtype='step', label='sin pred')\n",
    "ax[0].hist(truth[:,0], bins=n_bins, histtype='step', label='sin truth')\n",
    "ax[1].hist(pred[:,1] , bins=n_bins, histtype='step', label='cos pred')\n",
    "ax[1].hist(truth[:,1], bins=n_bins, histtype='step', label='cos truth')\n",
    "ax[2].hist(np.arccos(pred[:,1]), bins=n_bins, histtype='step' , label='Azimuth pred')\n",
    "ax[2].hist(np.arccos(truth[:,1]),bins=n_bins, histtype='step', label='Azimuth truth')\n",
    "\n",
    "ax[0].legend(loc='best')\n",
    "ax[1].legend()\n",
    "ax[2].legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.hist(xpred[:,1], histtype='step')\n",
    "plt.hist(xtruth[:,1], histtype='step');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

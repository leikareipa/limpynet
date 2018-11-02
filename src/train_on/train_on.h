/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 */

#ifndef TRAIN_ON_H
#define TRAIN_ON_H

class nnetwork_c;

// Called by main() to initialize the net for whatever data the user has. You'd
// implement this function specifically for the kind of data you have in mind.
// Returns true/false to reflect whether the function considers the initialization
// to have succeeded.
bool k_initialize_net_for_user_data(nnetwork_c *const net, const int argc, char *const argv[]);

// Gets called by main() to train the net on whatever data the user has. You'd
// implement this function specifically for the kind of data you have in mind.
// Returns true/false to reflect whether the function considers the training to
// have succeeded.
bool k_train_net_on_user_data(nnetwork_c *const net);

#endif

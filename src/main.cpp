/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 * A basic feedforward neural net, with a customizable layer structure.
 *
 */

#include <cstdlib>
#include "../src/nnetwork/nnetwork.h"
#include "../src/train_on/train_on.h"

int main(int argc, char *argv[])
{
    nnetwork_c net;

    if (!k_initialize_net_for_user_data(&net, argc, argv) ||
        !net.announce_current_configuration() ||
        !k_train_net_on_user_data(&net))
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

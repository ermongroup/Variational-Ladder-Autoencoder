import argparse
from visualize import *
import time

from dataset import *
from vladder import VLadder

class NoisyTrainer:
    def __init__(self, network, dataset, args):
        self.network = network
        self.dataset = dataset
        self.args = args
        self.batch_size = args.batch_size
        self.data_dims = self.dataset.data_dims
        self.train_with_mask = False
        self.train_discrete = False

        if args.plot_reconstruction:
            plt.ion()
            self.fig = plt.figure()
            plt.show()
            self.fig.suptitle("Reconstruction of " + str(self.network.name))
            self.network = network

        self.test_reconstruction_error = True

    def get_noisy_input(self, original):
        if not self.args.denoise_train:
            return original

        # Add salt and pepper noise
        noisy_input = np.multiply(original, np.random.binomial(n=1, p=0.9, size=[self.batch_size]+self.data_dims)) + \
                      np.random.binomial(n=1, p=0.1, size=[self.batch_size]+self.data_dims)

        # Add Gaussian noise
        noisy_input += np.random.normal(scale=0.1, size=[self.batch_size]+self.dataset.data_dims)

        # Activate following code to remove entire window of content. Not recommended
        # removed_width = random.randint(10, int(round(self.data_dims[0]/1.5)))
        # removed_height = random.randint(10, int(round(self.data_dims[1]/1.5)))
        # removed_left = random.randint(0, self.data_dims[0] - removed_width - 1)
        # removed_right = removed_left + removed_width
        # removed_top = random.randint(0, self.data_dims[1] - removed_height - 1)
        # removed_bottom = removed_top + removed_height
        # if random.random() > 0.5:
        #     noisy_input[:, removed_left:removed_right, removed_top:removed_bottom, :] = \
        #         np.zeros((self.batch_size, removed_width, removed_height, self.data_dims[-1]), dtype=np.float)
        # else:
        #     noisy_input[:, removed_left:removed_right, removed_top:removed_bottom, :] = \
        #         np.ones((self.batch_size, removed_width, removed_height, self.data_dims[-1]), dtype=np.float)

        return np.clip(noisy_input, a_min=self.dataset.range[0], a_max=self.dataset.range[1])

    def train(self):
        refresh_time = time.time()  # This is only a trick to prevent gui freeze for matplotlib

        # Visualization
        if self.network.do_generate_samples:
            sample_visualizer = SampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_conditional_samples:
            sample_visualizer_conditional = ConditionalSampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_manifold_samples:
            sample_visualizer_manifold = ManifoldSampleVisualizer(self.network, self.dataset)

        for iteration in range(10000000):
            iter_time = time.time()
            if time.time() - refresh_time > 0.2:
                plt.pause(0.001)
                refresh_time = time.time()

            images = self.dataset.next_batch(self.batch_size)
            noisy_input = self.get_noisy_input(images)
            train_loss = self.network.train(noisy_input, images)

            if iteration % 20 == 0:
                print("Iteration %d: Reconstruction loss %f, time per iter %fs" %
                      (iteration, train_loss, time.time() - iter_time))

            if iteration % 500 == 0:
                test_error = self.test(5)
                print("Reconstruction error @%d per pixel: " % iteration, test_error)

            if iteration % 1000 == 0:
                layers = [layer for layer in self.network.random_latent_code()]
                layers.sort()
                print("Visualizing %s" % layers)
                if self.network.do_generate_samples:
                    sample_visualizer.visualize(num_rows=10)
                if self.network.do_generate_conditional_samples:
                    sample_visualizer_conditional.visualize(layers=layers, num_rows=10)
                if self.network.do_generate_manifold_samples:
                    sample_visualizer_manifold.visualize(layers=layers, num_rows=30)

    """ Returns reconstruction error per pixel """
    def test(self, num_batch=3):
        error = 0.0
        for test_iter in range(num_batch):
            test_image = self.dataset.next_test_batch(self.batch_size)
            noisy_test_image = self.get_noisy_input(test_image)
            reconstruction = self.network.test(noisy_test_image)
            error += np.sum(np.square(reconstruction - test_image)) / np.prod(self.data_dims[:2]) / self.batch_size
            if test_iter == 0 and args.plot_reconstruction:
                # Plot the original image, noisy image, and reconstructed image
                self.plot_reconstruction(test_image, noisy_test_image, reconstruction)
        return error / num_batch

    def plot_reconstruction(self, test_image, noisy_image, reconstruction, num_plot=3):
        for img_index in range(num_plot):
            if self.data_dims[-1] == 1:
                self.fig.add_subplot(num_plot, 3, img_index*3+1).imshow(
                    self.dataset.display(test_image[img_index, :, :, 0]), cmap=plt.get_cmap('Greys'))
                self.fig.add_subplot(num_plot, 3, img_index*3+2).imshow(
                    self.dataset.display(noisy_image[img_index, :, :, 0]), cmap=plt.get_cmap('Greys'))
                self.fig.add_subplot(num_plot, 3, img_index*3+3).imshow(
                    self.dataset.display(reconstruction[img_index, :, :, 0]), cmap=plt.get_cmap('Greys'))
            else:
                self.fig.add_subplot(num_plot, 3, img_index*3+1).imshow(
                    self.dataset.display(test_image[img_index]))
                self.fig.add_subplot(num_plot, 3, img_index*3+2).imshow(
                    self.dataset.display(noisy_image[img_index]))
                self.fig.add_subplot(num_plot, 3, img_index*3+3).imshow(
                    self.dataset.display(reconstruction[img_index]))
        plt.draw()
        plt.pause(1)

# --dataset=svhn --denoise_train --plot_reconstruction --gpus=1 --db_path=dataset/svhn
# --dataset=celebA --denoise_train --plot_reconstruction --gpus=0 --db_path=/ssd_data/CelebA
# --dataset=mnist --gpus=2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--dataset', type=str, default='celebA')
    parser.add_argument('--netname', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--db_path', type=str, default='')
    parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                        help='Use denoise training by adding Gaussian/salt and pepper noise')
    parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                        help='Plot reconstruction')
    args = parser.parse_args()

    if args.gpus is not '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    plt.ion()
    plt.show()

    if args.dataset == 'mnist':
        dataset = MnistDataset()
    elif args.dataset == 'lsun':
        dataset = LSUNDataset(db_path=args.db_path)
    elif args.dataset == 'celebA':
        dataset = CelebADataset(db_path=args.db_path)
    elif args.dataset == 'svhn':
        dataset = SVHNDataset(db_path=args.db_path)
    else:
        print("Unknown dataset")
        exit(-1)
    model = VLadder(dataset, name=args.netname, batch_size=args.batch_size)
    trainer = NoisyTrainer(model, dataset, args)
    trainer.train()


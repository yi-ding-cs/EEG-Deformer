# This is the processing script of attention dataset
import argparse
import numpy as np
import os
import os.path as osp
from prepare_data import PrepareData
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)


class ATTEN(PrepareData):
    def __init__(self, args):
        super(ATTEN, self).__init__(args)
        self.num_ses = 3
        self.original_order = ['Fp1', 'AFF5', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3',
                               'Pz', 'POz', 'O1', 'Fp2', 'AFF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4',
                               'P8', 'O2', 'HEOG', 'VEOG']
        self.graph_fro = [['Fp1', 'AFF5'], ['AFF6', 'Fp2'],
                          ['F1', 'F2'],
                          ['AFz'],
                          ['FC5', 'FC1'], ['FC2', 'FC6'],
                          ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                          ['P7', 'P3', 'Pz', 'P4', 'P8'], ['POz'], ['O1', 'O2'],
                          ['T7'], ['T8']]
        self.graph_gen = [['Fp1', 'Fp2'], ['AFF5', 'AFz', 'AFF6'], ['F1', 'F2'],
                          ['FC5', 'FC1', 'FC2', 'FC6'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                          ['P7', 'P3', 'Pz', 'P4', 'P8'], ['POz'], ['O1', 'O2'],
                          ['T7'], ['T8']]
        self.graph_hem = [['Fp1', 'AFF5'], ['AFF6', 'Fp2'],
                          ['F1', 'F2'],
                          ['AFz', 'Cz', 'Pz', 'POz'],
                          ['FC5', 'FC1'], ['FC2', 'FC6'],
                          ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                          ['P7', 'P3'], ['P4', 'P8'], ['O1'], ['O2'],
                          ['T7'], ['T8']]
        self.TS = ['Fp1', 'AFF5', 'F1', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'O1',
                   'Fp2', 'AFF6', 'F2', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'O2']

        self.BL = ['Fp1', 'AFF5', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3',
                   'Pz', 'POz', 'O1', 'Fp2', 'AFF6', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4',
                   'P8', 'O2']

        self.filter_bank = [[4, 8], [8, 14], [14, 31]] if args.model in ['HRNN', 'GraphNet'] \
            else [[1, 4], [4, 8], [8, 12], [12, 30], [30, 45]]

        self.filter_allowance = [[2, 2], [2, 2], [2, 2]] if args.model in ['HRNN', 'GraphNet'] \
            else [[0.2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    def load_one_session(self, sub, ses):
        """
        This function load a certain session (ses) of the subject (sub)
        Args:
            sub: subject index
            ses: session index

        Returns:
            data: list of (time, chan)
        """
        #
        data = np.load(osp.join(self.data_path, 'sub{}_ses{}_eeg.npy'.format(sub, ses)),
                       allow_pickle=True)
        # get the label
        label = np.load(osp.join(self.data_path, 'sub{}_ses{}_label.npy'.format(sub, ses)),
                        allow_pickle=True)  # (trials,)
        # data: 12 x 30 x 4000
        # label: 12
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        # reorder the EEG channel to build the local-global graphs
        data = self.reorder_channel(data=data, graph_type=self.args.graph_type, graph_idx=self.BL)

        data_T = [trial.T for trial in data]   # list of (time x chan)

        return data_T, label

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load
        keep_dim: True for keeping the session dimension

        Returns
        -------
        data: (session, trial, time, chan) label: (session, trial)
        """
        data, label = [], []
        for ses in range(1, self.num_ses + 1):
            data_this_ses, label_this_ses = self.load_one_session(sub=sub, ses=ses)
            data.append(data_this_ses)
            label.append(label_this_ses)

        return data, label

    def create_dataset(self, subject_list, split=False, feature=False, band_pass_first=True, keep_session=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        sub_split: (bool) whether to split one segment's data into shorter sub-segment
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.pkl'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            if len(self.args.session_to_load) == 3:
                # we will use all three sessions
                if not keep_session:
                    data_temp, label_temp = [], []
                    for ses_idx in range(len(data_)):
                        data_temp.extend(data_[ses_idx])   # trial*ses, time, chan
                        label_temp.extend(label_[ses_idx])
                    data_ = data_temp
                    label_ = label_temp
            else:
                # use some sessions
                data_temp, label_temp = [], []
                for item in self.args.session_to_load:
                    data_temp.extend(data_[item - 1])
                    label_temp.extend(label_[item - 1])
                data_ = data_temp   # trial*selected ses, time, chan
                label_ = label_temp

            if band_pass_first:
                assert self.args.data_format not in ['PSD', 'rPSD'], "Please set band_pass_first=False if you want to" \
                                                                     "use PSD and rPSD as your features"
                data_ = self.get_filter_banks(
                    data=data_, fs=self.args.sampling_rate,
                    cut_frequency=self.filter_bank, allowance=self.filter_allowance
                )   # list of (time x chan x f)

            if split:
                data_, label_ = self.split_trial(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate,
                    sub_segment=self.args.sub_segment, sub_overlap=self.args.sub_overlap
                )

            if feature:
                # data_ : list of (segment, sequence, time, chan, f)
                data_ = self.get_features(
                    data=data_, feature_type=self.args.data_format
                )
            else:
                # data_ : list of (segment, time, chan)
                data_T = [np.transpose(item, (0, 2, 1)) for item in data_]
                data_ = data_T   # list of (segment, chan, time)

            print('Data and label for sub{} prepared!'.format(sub))
            self.save(data_, label_, sub-1)  # start with 0 instead of 1

    def load_original_data(self, path):
        # this function load the .eeg files
        raw = mne.io.read_raw_brainvision(path)
        raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog'})
        mne.rename_channels(raw.info, {'FP1': 'Fp1', 'FP2': 'Fp2'})

        montage = mne.channels.make_standard_montage(kind='standard_1005')
        raw.set_montage(montage)
        return raw

    def ICA_EOG_removal(self, raw, plot=False, filtering=True):
        # This function use EOG channels and ICA to remove EOGs
        raw.load_data()

        eog_evoked = create_eog_epochs(raw, ch_name=['HEOG', 'VEOG']).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))

        filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
        ica = ICA(n_components=15, max_iter='auto', random_state=97)
        ica.fit(filt_raw)

        ica.exclude = []
        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        ica.exclude = eog_indices

        if plot:
            raw.plot()
            eog_evoked.plot()
            # barplot of ICA component "EOG match" scores
            ica.plot_scores(eog_scores)

            # plot ICs applied to raw data, with EOG matches highlighted
            ica.plot_sources(raw, show_scrollbars=False)

            # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            ica.plot_sources(eog_evoked)
            ica.plot_sources(raw, show_scrollbars=False)
            # blinks
            ica.plot_overlay(raw, exclude=[0], picks='eeg')

        reconst_raw = raw.copy()
        ica.apply(reconst_raw)

        if plot:
            raw.plot()
            reconst_raw.plot()

        if filtering:
            reconst_raw = reconst_raw.filter(.5, None, method='iir')
            reconst_raw = reconst_raw.filter(None, 50., method='iir')

        return reconst_raw

    def get_epochs(self, raw):
        # This function helps to cut the continuous data into epochs (trials)
        events = mne.events_from_annotations(raw)
        events = events[0]

        # get event information of attention task
        events_attention = events.copy()

        # get event information of rest task
        events_rest = events.copy()
        idx_first_trial = np.where(events_rest[:, -1] == 48)[0][
            0]  # 48 is the start cule of one block (40s attention + 20s rest)
        events_rest[idx_first_trial][-1] = 0  # disable the start index of the first trial
        events_rest = np.concatenate((events_rest, np.array([[events_rest[-1][0] + 20 * 1000, 0, 48]])),
                                     axis=0)  # add the ending index of the last trial
        # get epochs according to the event codes
        '''
        Here we use the first half of the attention task to balance the data samples between attention 
        and rest as:
        Zhang, Yangsong, et al. "An end-to-end 3D convolutional neural network for decoding attentive mental state." 
        Neural Networks 144 (2021): 129-137.
        '''
        epochs_attention = mne.Epochs(raw, events_attention, event_id=[48], tmin=0.0, tmax=20.0,
                                      baseline=None)
        epochs_rest = mne.Epochs(raw, events_rest, event_id=[48], tmin=-20.0, tmax=0.0,
                                 baseline=None)
        # load the data and do resampling
        epochs_attention.load_data()
        epochs_rest.load_data()
        epochs_attention = epochs_attention.resample(sfreq=200)
        epochs_rest = epochs_rest.resample(sfreq=200)

        # get the data
        data_attention = epochs_attention.get_data()
        data_rest = epochs_rest.get_data()
        return data_attention, data_rest

    def decode_data(self, num_ses, num_subject, path, name_dataset):
        # loop each session
        for ses in range(1, num_ses + 1):
            # loop each subject
            for sub in range(1, num_subject + 1):
                if sub <= 9:
                    sub_code = '0{}'.format(sub)
                else:
                    sub_code = sub

                path_load = os.path.join(path, 'VP0{}'.format(sub_code), 'gonogo{}.vhdr'.format(ses))
                raw = self.load_original_data(path_load)
                # plot_eeg(raw.get_data()[:, :10000])
                reconst_raw = self.ICA_EOG_removal(raw=raw, plot=False, filtering=True)
                # plot_eeg(reconst_raw.get_data()[:, :10000])
                # plot_eeg(data[-1])
                data_att, data_rest = self.get_epochs(raw=reconst_raw)
                data = np.concatenate((data_att, data_rest), axis=0)
                label = np.array([1, 0]).repeat(data_att.shape[0])
                # plot_eeg(data[0])
                # plot_eeg(data[-1])
                save_path = osp.join(self.save_path, 'decoded_{}'.format(name_dataset))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    pass
                np.save(osp.join(save_path, 'sub{}_ses{}_label'.format(sub, ses)), label)
                np.save(osp.join(save_path, 'sub{}_ses{}_eeg'.format(sub, ses)), data)
                print('Sub{} Ses{} is saved'.format(sub, ses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--ROOT', type=str, default=osp.join(os.getcwd(), '..'))
    parser.add_argument('--dataset', type=str, default='ATTEN')
    parser.add_argument('--data-decoded', type=bool, default=True)
    parser.add_argument('--data-path', type=str, default=osp.join(os.getcwd(), '..', 'data_processed', 'decoded_ATTEN'))
    parser.add_argument('--raw-data-path', type=str, default='D:\\DingYi\\Dataset\\ThreeCognitiveTasks')
    parser.add_argument('--subjects', type=int, default=26)
    parser.add_argument('--session-to-load', default=[1, 2, 3])
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='ATTEN', choices=['ATTEN'])
    parser.add_argument('--segment', type=int, default=4)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--sampling-rate', type=int, default=200)
    parser.add_argument('--input-shape', type=tuple, default=(1, 28, 800))
    parser.add_argument('--data-format', type=str, default='eeg', choices=['DE', 'Hjorth', 'PSD', 'rPSD', 'rpower',
                                                                           'sta', 'multi-view', 'raw', 'power', 'eeg'])
    parser.add_argument('--graph-type', type=str, default='BL', choices=['aff', 'gen', 'hem', 'TS', 'BL'])
    parser.add_argument('--model', type=str, default='ECENet')
    parser.add_argument('--sub-segment', type=int, default=0, help="Window length of each time sequence")  # 2 for EmT
    parser.add_argument('--sub-overlap', type=float, default=0.0)

    args = parser.parse_args()
    atten = ATTEN(args)
    if not args.data_decoded:
        atten.decode_data(num_ses=3, num_subject=args.subjects, path=args.raw_data_path, name_dataset=args.dataset)
    sub_to_load = np.arange(1, 27)
    atten.create_dataset(
        subject_list=sub_to_load, split=True,
        feature=False if args.data_format in ['eeg'] else True,
        band_pass_first=False if args.data_format in ['PSD', 'rPSD', 'eeg'] else True,
        keep_session=False
    )








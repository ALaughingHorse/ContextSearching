from tokenizer_xm import *
import pandas as pd
import numpy as np


class context_searching:

    """
    Search the context of a specific token.

    ---Parameters:
    target_token: the token we are looking at

    doc: a list/array of reviews (preferably containing the target_token but not required)

    left_window: an integer indicating the span to the left of the target token in each text body of doc where
    we start looking for context

    right_window: an integer indicating the span to the right of the target token in each text body of doc where
    we stop looking for context

    padding_token: If the length of the text body is less than (left_window + right_window + 1), fill the resulting
    Context Probing Matrix with this token

    """

    
    
    def __init__(self, target_token, doc, left_window = 10,right_window = 10,padding_token = "_empty_"):
        """
        Initializing all the variables
        """
        self.target_token = target_token
        self.doc = doc
        self.left_window = left_window
        self.right_window = right_window
        self.padding_token = padding_token

         # Define supporting functions
        def get_word_location(token, doc):
            """
            Takes in a token and a list of tokens, output the locations of the token
            """
            doc = np.array(doc)

            return np.array(range(len(doc)))[(doc == token)]
        
        def get_neighbours(target_loc, doc, left_window, right_window,padding_token):

            """
            Takes in a list of tokens, a location and a left-right window. The function will return the tokens spanning \
            the left-right window of the target location. The function also automatically pad the sequence if there is \
            insufficient records within the boundaries specified by the windows

            ---Parameters:
            target_loc: target location
            doc: list of of tokens
            """

            doc = list(doc)
            # get the right boundary
            right_bound = len(doc)

            # get the left tokens
            dif_left = target_loc - left_window  
            ## if dif_left is smaller than zero
            if dif_left < 0:
                pad = [padding_token] * abs(dif_left)
                left_tokens = pad + doc[0:target_loc]

            else:
                left_tokens = doc[(target_loc - left_window):target_loc]

            assert len(left_tokens) == left_window

            # get the right tokens
            right_end = target_loc + right_window

            # if right_end is greater than right bound
            if right_end >= right_bound:
                pad = [padding_token] * (1 + abs(right_end - right_bound))
                right_tokens = doc[(target_loc+1):(right_bound + 1)] + pad

            else:
                right_tokens = doc[(target_loc+1):(right_end + 1)]
            assert len(right_tokens) == right_window


            return left_tokens + [doc[target_loc]] + right_tokens

        def update_dict(vocab, key, val):
            """
            Update the dictionary vocab with new key and value. If key already exists, append new value
            """

            if key in vocab:
                vocab.update({key:vocab[key] + [val]})
            else:
                vocab.update({key:[val]})

            return vocab

        # Function defined above
        self.update_dict = update_dict
        self.get_word_location = get_word_location
        self.get_neighbours = get_neighbours
    

    def get_context_prob_matrix(self,stop_words = [],lemmatize = True, stem = False):
        """
        Construct the context probing matrix. 
        stop_words: a list of words to exclude in text pre-processing
        lemmatize: a boolean value indicating whether the tokens should be lemmatized, default to True
        stem: a boolean value indicating whether the tokens shoudl be stemmed, default to False
        """
        # Inherit variables from init
        left_window = self.left_window
        right_window = self.right_window
        
        # Pre-processing the docs and tokens
        
        ## pre_process the tokens
        tk = text_tokenizer_xm(text = self.doc,lemma_flag = lemmatize, stem_flag = stem,   stopwords=stop_words)
        docs = tk.txt_pre_pros_all()
        ## Make sure the docs is a list
        docs = list(docs)
        
        ## pre_processed the target token
        processed_target = text_tokenizer_xm(text = self.target_token,stopwords=stop_words).txt_pre_pros()[0]
        
        
        # initializing
        position = list(range(-left_window,right_window + 1))
        vocab = dict()
        cpm = []

        for doc in docs:
            # Get the location of the target token
            locs = self.get_word_location(token = processed_target, doc = doc)

            if len(locs) == 0:
                cpm.append([self.padding_token] * len(position))
            else:
                loc = min(locs) # here if there is multiple, picked the first occurance

                # Get the tokens
                tks = self.get_neighbours(target_loc = loc, doc = doc, left_window = left_window, right_window=right_window, \
                                     padding_token = self.padding_token)

                # Fill the cpm
                cpm.append(tks)

                # Fill the dic
                for idx in range(len(position)):
                    vocab = self.update_dict(vocab = vocab,key = tks[idx],val = position[idx])

        # Assign the results to self
        self.context_prob_matrix = cpm
        self.vocabs = vocab
        self.position_idx = position

        class cpm_output:
            
            # Get the results
            vocabs = vocab
            context_prob_matrix = cpm
            position_idx = position
                
            def get_cpm_stats_tb():
                """
                Compute the stats from context probing matrix
                """
                
                # initialize the lists
                tokens = []
                means = []
                variance = []
                abs_means = []
                # how many times a term appears within the window
                count = []
                medians =[]

                for key, value in self.vocabs.items():
                    if key not in ["_empty_"]: # excluding the padding token
                        tokens += [key]
                        means += [np.mean(value)]
                        abs_means += [np.mean([abs(x) for x in value])]
                        variance += [np.var(value)]
                        count += [len(value)]
                        medians += [np.median(value)]
                        
                cpm_stats_tb = pd.DataFrame({"tokens":tokens,"mean":means,"variance":variance,\
                                            "abs_mean":abs_means,"count":count,"median":medians})
                
                self.cpm_stats_tb = cpm_stats_tb
                
                class cpm_stats_tb_class():
                    cpm_stats_tb = self.cpm_stats_tb 
                    
                    def guess_ngram(n = 2):
                        """
                        Based on CPM stats tb, construct the Ngrams
                        """
                         # Get the unique indexes
                        uniq_indx = np.array(range(-(n-1),n))

                        # For each indx in uniq_indx, look up the cm_stats_tb and record the max count
                        max_count = []
                        best_tokens = []

                        for idx in uniq_indx:
                            temp_tb = cpm_stats_tb.loc[cpm_stats_tb['median'] == idx,:].sort_values('count',ascending = False)[['tokens','count']].reset_index(drop = True)
                            max_count.append(temp_tb['count'][0])
                            best_tokens.append(temp_tb['tokens'][0])

                        # Next decide the index window
                        idx_windows = []
                        for i in range(0,n):
                            idx_windows.append(list(range(i,i+ n)))

                        # Get the ngram candidates
                        ngram_candidate = []
                        total_score = []
                        for win in idx_windows:
                            win_array = np.array(win)
                            ngram_candidate.append(" ".join(np.array(best_tokens)[win_array]))
                            total_score.append(sum(np.array(max_count)[win_array]))

                        output_df = pd.DataFrame({'ngram_candidates':ngram_candidate,\
                                                 "total_scores":total_score})
                           
                        return output_df.sort_values('total_scores',ascending = False).reset_index(drop = True)
                        
                
                return cpm_stats_tb_class
       
        return cpm_output
    
    
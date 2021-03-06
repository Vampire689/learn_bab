import torch
from plnn.proxlp_solver.propagation import Propagation
from nn_learning.nn_branching.network import ExpNet
import math
import torch.nn.functional as F
from torch import optim
import random

class BranchingChoice:
    """
    Class implementing branching decisions. The branching is accessed via the branch function.
    """

    def __init__(self, branch_dict, layers, sparsest_layer=0):
        """
        :param branch_dict: dictionary containing the settings for the brancher
        :param layers: list of pytorch NN layers describing the network to perform the branching on
        :param sparsest_layer: if all layers are dense, set it to -1
        """
        from tools.bab_tools.bab_runner import TRAIN, branch_model_path
        # Store content of branch_dict as class attributes.
        for key in ['heuristic_type', 'bounding', 'max_domains']:
            if key == "bounding":
                self.__dict__[key] = branch_dict[key]["net"]
            else:
                self.__dict__[key] = branch_dict[key]
        if self.heuristic_type not in ["SR", "FSB", "input", "NN"]:
            raise NotImplementedError(f"Branching heuristic {self.heuristic_type} not implemented.")

        # Set other parameters.
        self.layers = layers
        self.sparsest_layer = sparsest_layer
        self.device = layers[0].weight.device
        self.train = TRAIN and (self.heuristic_type in ["NN"])
        self.nn_dict = branch_model_path

        # Set SR-specific parameters and variables.
        if self.heuristic_type in ["SR", "FSB", "NN"]:
            self.icp_score_counter = 0
            # if the maximum score is below the threshold, we consider it to be non-informative
            self.decision_threshold = 0.001

        # init NN model
        if self.heuristic_type in ["NN"]:
            self.model = ExpNet(64).to(self.device)
            if self.train:
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, eps=1e-5, weight_decay=1e-4)
                try:
                    checkpoint = torch.load(self.nn_dict)    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded previous model and optimizer")
                except:    pass
                self.model.train()
            else: 
                self.model.load_state_dict(torch.load(self.nn_dict)['model_state_dict'])
                self.model.eval()

        # Set the branching function.
        branching_function_dict = {"SR": self.branch_sr, "FSB": self.branch_fsb, "input": self.branch_input_bab, "NN": self.branch_nn}
        self.branching_function = branching_function_dict[self.heuristic_type]

        # Set the splitting function.
        splitting_function_dict = {"SR": self.relu_split, "FSB": self.relu_split, "input": self.input_split, "NN": self.relu_split}
        self.splitting_function = splitting_function_dict[self.heuristic_type]

    def branch(self, domains, lower_bounds, upper_bounds):
        # Given batches of domains, lower and upper bounds, compute and return a list of branching decision for
        # each of them.
        return self.branching_function(domains, lower_bounds, upper_bounds)

    def babsr_scores(self, domains, lower_bounds, upper_bounds, minsplit=True):
        '''
        Score each neuron's splitting influence based on each node's contribution to the cost function in the KW
        formulation.
        Operates on a batches of intermediate bounds and relu masks.
        'minsplit' uses the min splitting advantage instead of the max
        Returns the list of scores (tensors) and a back-up scoring (a subset of the true score).
        '''
        # Compute KW dual variables for the batched lower bounding of the networks's only output.
        prop_net = Propagation(self.layers, type="KW")
        prop_net.build_model_using_bounds(domains, (lower_bounds, upper_bounds))
        kw_vars = prop_net.get_closedform_lastlayer_lb_duals()

        # Compute the BaBSR scores, see https://arxiv.org/abs/2104.06718, section 4.1
        # The following computations rely on the unconditioned bias for the first layer
        # (see build_first_conditioned_layer), as the absence of W_1 (first layer weights) in the BaBSR scores makes
        # the conditioning on the b_1 bias only incorrect.
        score_s, score_t, mask, nonambscores = [], [], [], []
        batch_size = lower_bounds[0].shape[0]
        for lay, (clbs, cubs) in enumerate(zip(lower_bounds, upper_bounds)):
            if lay > 0 and lay < len(lower_bounds) - 1:
                cmask = ((cubs > 0) & (clbs < 0)).unsqueeze(1)
                bias = - ((clbs * cubs) / (cubs - clbs)).unsqueeze(1).masked_fill_(~cmask, 0)
                intercept_score = bias * kw_vars.lambdas[lay - 1].clamp(0, None)
                cscore = (kw_vars.lambdas[lay - 1] * prop_net.weights[lay - 1].get_unconditioned_bias())
                if minsplit:
                    # using the min splitting advantage instead of the max has a large effect on performance
                    cscore.clamp_(None, 0)
                else:
                    cscore.clamp_(0, None)
                cscore -= kw_vars.mus[lay - 1] * prop_net.weights[lay - 1].get_unconditioned_bias() + intercept_score
                cscore = torch.abs(cscore) * cmask
                nonambscore = torch.abs(kw_vars.mus[lay - 1])

                score_s.append(cscore.reshape((batch_size, -1)))
                score_t.append(-intercept_score.reshape((batch_size, -1)))
                mask.append(cmask.reshape((batch_size, -1)))
                nonambscores.append(nonambscore.reshape((batch_size, -1)))

                if torch.isnan(cscore).any() or torch.isnan(intercept_score).any():
                    raise ValueError("NaN branching score detected.")

        return score_s, score_t, mask, nonambscores, prop_net.weights

    def branch_sr(self, domains, lower_bounds, upper_bounds):
        '''
        choose the dimension to split on
        based on each node's contribution to the cost function in the KW formulation.
        '''
        batch_size = lower_bounds[0].shape[0]
        score, intercept_tb, mask, nonamb, net_weights = self.babsr_scores(
            domains, lower_bounds, upper_bounds, minsplit=False)

        # priority to each layer when making a random choice with preferences. Increased preference for later elements
        # in the list
        random_order = list(range(len(net_weights) - 1))
        try:
            random_order.remove(self.sparsest_layer)
            random_order = [self.sparsest_layer] + random_order
        except:
            pass

        # The following is not parallelised (as in the original BaBSR implementation)
        decision_list = []
        all_set_idcs = []
        for idx in range(batch_size):
            random_choice = random_order.copy()
            mask_item = [m[idx] for m in mask]
            score_item = [s[idx] for s in score]
            max_info = [torch.max(i, 0) for i in score_item]
            decision_layer = max_info.index(max(max_info))
            decision_index = max_info[decision_layer][1].item()
            if decision_layer != self.sparsest_layer and max_info[decision_layer][0].item() > self.decision_threshold:
                decision = [decision_layer, decision_index]

            else:
                intercept_tb_item = [i_tb[idx] for i_tb in intercept_tb]
                min_info = [[i, torch.min(intercept_tb_item[i], 0)] for i in range(len(intercept_tb_item)) if
                            torch.min(intercept_tb_item[i]) < -1e-4]
                if len(min_info) != 0 and self.icp_score_counter < 2:
                    intercept_layer = min_info[-1][0]
                    intercept_index = min_info[-1][1][1].item()
                    self.icp_score_counter += 1
                    decision = [intercept_layer, intercept_index]
                    if intercept_layer != 0:
                        self.icp_score_counter = 0
                    # print('\tusing intercept score')
                else:
                    # print('\t using a random choice')
                    undecided = True
                    while undecided:
                        if len(random_choice) > 0:
                            preferred_layer = random_choice.pop(-1)
                            if len(mask_item[preferred_layer].nonzero()) != 0:
                                decision = [preferred_layer, mask_item[preferred_layer].nonzero()[0].item()]
                                undecided = False
                            else:
                                pass
                        else:
                            # all ReLUs have been set
                            decision = [None, None]
                            all_set_idcs.append(idx)
                            break
                    self.icp_score_counter = 0
            decision_list.append(decision)

            # TODO: should run an LP solver instead
            # ReLUs have all been set, use a scoring for the ambiguous ReLUs.
            if len(all_set_idcs) > 0:
                for idx in all_set_idcs:
                    nonamb_item = [s[idx] for s in score]
                    nonamb_max_info = [torch.max(i, 0) for i in nonamb_item]
                    dec_layer = max_info.index(max(nonamb_max_info))
                    decision_list[idx] = [dec_layer, nonamb_max_info[dec_layer][1].item()]

        return decision_list

    def branch_fsb(self, domains, lower_bounds, upper_bounds):
        """
        Choose which neuron to split on by evaluating the contribution to the final layer bounding of fixing
        the most promising neurons according to the KW heuristic.
        :param max_domains: if not None, indicates how many domains to compute at once with bounding_net
        """

        kw_score, intercept_score, mask, nonamb, net_weights = self.babsr_scores(domains, lower_bounds, upper_bounds, minsplit=True)
        primal_score = kw_score
        secondary_score = [-cintercept for cintercept in intercept_score]

        # [1,1] means that we consider the single best score per layer.
        fsb_decisions, impr = self.filtered_branching(
            domains, lower_bounds, upper_bounds, [primal_score, secondary_score], [1, 1])

        # Use SR as fallback strategy when FSB would re-split.
        if (impr == 0).any():
            sr_counter = 0
            sr_decisions = self.branch_sr(domains, lower_bounds, upper_bounds)
            entries_to_check = (impr == 0).nonzero()
            for batch_idx in entries_to_check:
                clb = lower_bounds[fsb_decisions[batch_idx][0] + 1][batch_idx].view(-1)[fsb_decisions[batch_idx][1]]
                cub = upper_bounds[fsb_decisions[batch_idx][0] + 1][batch_idx].view(-1)[fsb_decisions[batch_idx][1]]
                if cub <= 0 or clb >= 0:
                    fsb_decisions[batch_idx] = sr_decisions[batch_idx]
                    sr_counter += 1
            print(f"Using BaBSR for {sr_counter} decisions")

        return fsb_decisions

    def filtered_branching(self, domains, lower_bounds, upper_bounds, branch_scores_lists, n_cands_per_layer=[1,1]):
        """
            Given list of lists of scores branch_scores_lists (and how many candidates per layer to pick from each list,
             passed as a list), estimate the bounding gain of splitting on the best scores per layer with the bounding
             algorithm represented by self.bounding
        """
        cand_per_layer = sum(n_cands_per_layer)
        dic = {1:0, 2:0, 3:1, 4:2, 5:2, 6:3, 7:3, 8:3, 9:3}
        n_layers = len(branch_scores_lists[0]) if not self.train else (len(branch_scores_lists[0])-dic[max(n_cands_per_layer)])

        for x_idx in range(n_layers):
            csplitcand_list = []
            for lst_idx, scores_set in enumerate(branch_scores_lists):
                # scores_list are indexed with an offset if doing domains
                csplitcand_list.append(torch.topk(scores_set[x_idx], n_cands_per_layer[lst_idx])[1])
            if x_idx == 0:
                split_candidates = torch.cat(csplitcand_list, 1).unsqueeze(0)
            else:
                cidxs = torch.cat(csplitcand_list, 1).unsqueeze(0)
                split_candidates = torch.cat([split_candidates, cidxs], 0)

        batch_size = lower_bounds[0].shape[0]
        repeat_factor = 1 + 2 * cand_per_layer * n_layers
        split_idx_offset = cand_per_layer * n_layers

        # Create the ranges for batches related to lower and upper splitting upfront for ease of coding.
        index_instr = [None] * repeat_factor
        index_instr[0] = (None, None, None)
        for lay_idx in range(n_layers):
            for cneuron in range(cand_per_layer):
                low_batch = 1 + cand_per_layer * lay_idx + cneuron
                upp_batch = low_batch + split_idx_offset

                # given a range of batch indices, I need to know which lay_idx and c_neurons to split on
                index_instr[low_batch] = (lay_idx, cneuron, "LB")
                index_instr[upp_batch] = (lay_idx, cneuron, "UB")

        doms_size = repeat_factor * batch_size
        # make max_domains a multiple of batch_size
        effective_doms = min((self.max_domains // batch_size) * batch_size, doms_size) if self.max_domains is not None \
            else doms_size
        assert effective_doms > 0, "max_domains can't be smaller than batch size"
        n_batches = int(math.ceil(doms_size / float(effective_doms)))

        branch_lbs = None
        for sub_batch_idx in range(n_batches):
            # compute bounds on sub-batch
            start_batch_index = sub_batch_idx * effective_doms // batch_size
            end_batch_index = min((sub_batch_idx + 1) * effective_doms, doms_size) // batch_size

            # The stacks are: single entry for benchmark bounds, then one entry for each neuron to split of each layer.
            # The first half of this splits on lower bounds, the second on upper bounds.
            c_repeat_factor = end_batch_index - start_batch_index
            lbs_stack = [clbs.repeat((c_repeat_factor,) + (1,) * (clbs.dim() - 1)) for clbs in lower_bounds]
            ubs_stack = [cubs.repeat((c_repeat_factor,) + (1,) * (cubs.dim() - 1)) for cubs in upper_bounds]
            domains_stack = domains.repeat((c_repeat_factor,) + (1,) * (domains.dim() - 1))
            for cbatch, (lay_idx, cneuron, btype) in enumerate(index_instr[start_batch_index:end_batch_index]):
                if lay_idx is None:
                    continue
                stack_range = list(range(cbatch * batch_size, (cbatch + 1) * batch_size))
                if btype in ["LB", "UB"]:
                    # IBs are indexed with an offset if doing domains
                    ib_idx = lay_idx + 1

                    clbs = lbs_stack[ib_idx][stack_range].view(batch_size, -1).\
                        gather(1, split_candidates[lay_idx][:, cneuron].unsqueeze(1))
                    cubs = ubs_stack[ib_idx][stack_range].view(batch_size, -1). \
                        gather(1, split_candidates[lay_idx][:, cneuron].unsqueeze(1))
                    # Split ambiguous ReLUs at 0 so as to get two linear functions, halve domains of non-ambiguous ReLUs
                    split_point = torch.where((clbs < 0) & (cubs > 0), torch.zeros_like(clbs), (clbs + cubs)/2)
                    if btype == "LB":
                        # Lower split (lbs_stack[lay_idx+1][stack_range] returns a copy).
                        lbs_stack[ib_idx][stack_range] = lbs_stack[ib_idx][stack_range].view(batch_size, -1).\
                            scatter(1, split_candidates[lay_idx][:, cneuron].unsqueeze(1), split_point). \
                            view_as(lbs_stack[ib_idx][stack_range])
                    else:
                        # "UB"
                        ubs_stack[ib_idx][stack_range] = ubs_stack[ib_idx][stack_range].view(batch_size, -1).\
                            scatter(1, split_candidates[lay_idx][:, cneuron].unsqueeze(1), split_point). \
                            view_as(ubs_stack[ib_idx][stack_range])

            # Bounding computation.
            self.bounding.build_model_using_bounds(domains_stack, (lbs_stack, ubs_stack))
            c_branch_lbs = self.bounding.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
            self.bounding.unbuild()

            branch_lbs = c_branch_lbs if branch_lbs is None else torch.cat([branch_lbs, c_branch_lbs], 0)

        branch_lbs = branch_lbs.squeeze(-1)
        baseline = branch_lbs[:batch_size]
        branch_lbs = branch_lbs[batch_size:].view(2, n_layers, cand_per_layer, batch_size)
        scores = ((branch_lbs - baseline) / torch.abs(baseline)).min(0)[0]  # the min between the LB/UB splits performs better than the average
        scores = torch.where(torch.isnan(scores), -1 * torch.ones_like(scores), scores)
        scores = torch.clamp(scores, min=0, max=1)
        max_per_layer, ind_per_layer = torch.max(scores, 1)
        impr, layer_ind = torch.max(max_per_layer, 0)

        selector = torch.arange(split_candidates.shape[1], device=split_candidates.device). \
            view(split_candidates.shape[1], 1)
        neuron_indices = ind_per_layer[layer_ind].gather(1, selector)
        gathered_candidates = split_candidates[layer_ind].gather(1, selector.unsqueeze(-1).expand(
            selector.shape + (split_candidates[layer_ind].shape[2],))).squeeze(1)
        neuron_indices = gathered_candidates.gather(1, neuron_indices).squeeze(1).tolist()
        branching_decision_list = list(zip(layer_ind.tolist(), neuron_indices))

        if self.train:
            return branching_decision_list, impr, split_candidates, scores.transpose(1,2)
        else:
            return branching_decision_list, impr

    def branch_nn(self, domains, lower_bounds, upper_bounds):
        batch_size = lower_bounds[0].shape[0]
        kw_score, intercept_score, mask, nonamb, net_weights = self.babsr_scores(
            domains, lower_bounds, upper_bounds, minsplit=False)
        primal_score = kw_score
        secondary_score = [-cintercept for cintercept in intercept_score]

        if self.train:
            self.model.train()
            self.optimizer.zero_grad()
            with torch.enable_grad():
                nn_score = self.model(lower_bounds, upper_bounds, mask, self.layers, primal_score, secondary_score)

                # generate ground truth
                layer_cands = random.randint(1, 7)
                decisions, impr, split_candidates, fsb_score = self.filtered_branching(
                    domains, lower_bounds, upper_bounds, [primal_score, secondary_score], [layer_cands, layer_cands])
                half = split_candidates.shape[2]//2
                gt_score_idx1, gt_score_idx2 = split_candidates[:, :, :half], split_candidates[:, :, half:]
                gt_score1, gt_score2 = fsb_score[:, :, :half], fsb_score[:, :, half:]

                # optimize NN
                loss = 0
                for i in range(len(gt_score_idx1)):
                    nn_candi_score1 = nn_score[i][:,:,0].gather(dim=1, index=gt_score_idx1[i])
                    nn_candi_score2 = nn_score[i][:,:,1].gather(dim=1, index=gt_score_idx2[i])
                    loss += F.l1_loss(nn_candi_score1, gt_score1[i]) + F.l1_loss(nn_candi_score2, gt_score2[i])
                loss.backward()
                self.optimizer.step()
                print('loss:', loss.data)
                torch.cuda.empty_cache()
        else:
            self.model.eval()
            with torch.no_grad():
                nn_score = self.model(lower_bounds, upper_bounds, mask, self.layers, primal_score, secondary_score)
            
            decisions = []
            impr = torch.zeros(batch_size).to(self.device)
            for idx in range(batch_size):
                score_item = [torch.max(s[idx], 1)[0] for s in nn_score]
                max_info = [torch.max(i, 0) for i in score_item]
                max_score = max(max_info)
                decision_layer = max_info.index(max_score)
                decision_index = max_info[decision_layer][1].item()
                decisions.append([decision_layer, decision_index])
                impr[idx] = max_score.values

        # Use SR as fallback strategy when FSB would re-split.
        if (impr == 0).any():
            sr_counter = 0
            sr_decisions = self.branch_sr(domains, lower_bounds, upper_bounds)
            entries_to_check = (impr == 0).nonzero()
            for batch_idx in entries_to_check:
                clb = lower_bounds[decisions[batch_idx][0] + 1][batch_idx].view(-1)[decisions[batch_idx][1]]
                cub = upper_bounds[decisions[batch_idx][0] + 1][batch_idx].view(-1)[decisions[batch_idx][1]]
                if cub <= 0 or clb >= 0:
                    decisions[batch_idx] = sr_decisions[batch_idx]
                    sr_counter += 1
            print(f"Using BaBSR for {sr_counter} decisions")

        return decisions

    def save_nn(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.nn_dict)

    def split_subdomain(self, decision, choice, batch_idx, domains, lbs_stacks, ubs_stacks):
        """
        Given a branching decision and stacks of bounds for all the activations, duplicated so that the i-th
        subdomain has copies at entries 2*i, 2*i + 1 clip the bounds according to the decision.
        Update performed in place in the list of lower/upper bound stacks (batches of lower/upper bounds)
        :param decision: tuples (x_idx, node) indicating where the split is performed
        :param choice: 0/1 for whether to clip on a blocking/passing ReLU
        :param domains: batched domains to update with the splitted ones at batch_idx
        :param lbs_stacks: batched lower bounds to update with the splitted ones at batch_idx
        :param ubs_stacks: batched upper bounds to update with the splitted ones at batch_idx
        """
        self.splitting_function(decision, choice, batch_idx, domains, lbs_stacks, ubs_stacks)

    @staticmethod
    def relu_split(decision, choice, batch_idx, domains, splitted_lbs_stacks, splitted_ubs_stacks):
        """
        Given a branching decision, perform ReLU splitting (sets the pre-activation LB to 0 to enforce a passing ReLU,
        the pre-activation UB to 0 to enforce a blocking ReLU).
        """
        if decision is not None:
            change_idx = decision[0] + 1
            is_ambiguous = (splitted_lbs_stacks[change_idx][batch_idx].view(-1)[decision[1]] < 0 <
                            splitted_ubs_stacks[change_idx][batch_idx].view(-1)[decision[1]])
            half_point = (splitted_lbs_stacks[change_idx][batch_idx].view(-1)[decision[1]] +
                          splitted_ubs_stacks[change_idx][batch_idx].view(-1)[decision[1]]) / 2
            # Split ambiguous ReLUs at 0 so as to get two linear functions, halve the domains of non-ambiguous ReLUs.
            split_point = 0 if is_ambiguous else half_point
            if not is_ambiguous:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Resplitting")
            if choice == 0:
                # blocking ReLU obtained by setting the pre-activation UB to 0
                splitted_ubs_stacks[change_idx][batch_idx].view(-1)[decision[1]] = split_point
            else:
                # passing ReLU obtained by setting the pre-activation LB to 0
                splitted_lbs_stacks[change_idx][batch_idx].view(-1)[decision[1]] = split_point

    def branch_input_bab(self, domains, lower_bounds, upper_bounds):
        # do input splitting along max dimension
        split_indices = torch.argmax((domains.select(-1, 1) - domains.select(-1, 0)).view(lower_bounds[0].shape[0], -1),
                                     dim=-1)
        return [(-1, csplit) for csplit in split_indices]

    @staticmethod
    def input_split(decision, choice, batch_idx, domains, splitted_lbs_stacks, splitted_ubs_stacks):
        if decision is not None:
            half_point = ((domains.select(-1, 1).view(splitted_lbs_stacks[0].shape[0], -1)[batch_idx, decision[1]] +
                           domains.select(-1, 0).view(splitted_lbs_stacks[0].shape[0], -1)[batch_idx, decision[1]]) / 2)
            # Split ambiguous ReLUs at 0 so as to get two linear functions, halve the domains of non-ambiguous ReLUs.
            if choice == 0:
                # blocking ReLU obtained by setting the pre-activation UB to 0
                domains.select(-1, 1).view(splitted_lbs_stacks[0].shape[0], -1)[batch_idx, decision[1]] = half_point
            else:
                # passing ReLU obtained by setting the pre-activation LB to 0
                domains.select(-1, 0).view(splitted_lbs_stacks[0].shape[0], -1)[batch_idx, decision[1]] = half_point

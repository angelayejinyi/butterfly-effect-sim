"""Offline checks for paper configurations, graph controls, and model requests."""
import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import EC_experiment as ec
import run_experiments as runs
import networkx as nx


class ExperimentTests(unittest.TestCase):
    def test_full_plan(self):
        args = argparse.Namespace(dimension="all", model_family="claude", model=None,
            profile="paper", seeds=list(range(30)), workers=4,
            personas=runs.EC_ROOT / "inputs/ai_regulation_minimal_personas_100.json",
            network_root=runs.EC_ROOT / "inputs/networks", out_root=Path('/tmp/ec-plan-test'))
        plan = runs.build_plan(args)
        self.assertEqual(len(plan), 450)
        self.assertEqual(len({p['out_dir'] for p in plan}), 450)
        self.assertTrue(all(Path(p['edgelist_path']).is_file() for p in plan))
        activation = [p for p in plan if '/activation/' in p['out_dir']]
        self.assertEqual({p['activation_prob'] for p in activation}, {.3,.5})
        self.assertEqual({p['seed'] for p in activation}, set(range(30)))
        args.profile='recorded'
        activation = [p for p in runs.build_plan(args) if '/activation/' in p['out_dir']]
        self.assertEqual({p['activation_prob'] for p in activation}, {.3,.5})

    def test_graph_bands_and_degrees(self):
        personas = ec.load_personas(str(runs.EC_ROOT/'inputs/ai_regulation_minimal_personas_100.json'))
        degree_maps=[]
        for (_, overrides), band in zip(runs.conditions('homophily','gpt','paper'),[(.02,.10),(.12,.20),(.22,.30)]):
            graph = ec.load_graph(str(runs.EC_ROOT/'inputs/networks'/overrides['network']),personas)
            value=nx.numeric_assortativity_coefficient(graph,'initial_stance')
            self.assertTrue(band[0] <= value <= band[1],value)
            degree_maps.append(dict(graph.degree()))
        self.assertEqual(degree_maps[0],degree_maps[1])
        self.assertEqual(degree_maps[1],degree_maps[2])
        for _,overrides in runs.conditions('hub_assignment','gpt','paper'):
            graph=ec.load_graph(str(runs.EC_ROOT/'inputs/networks'/overrides['network']),personas)
            self.assertEqual(sorted(dict(graph.degree()).values()),sorted(degree_maps[0].values()))
            self.assertTrue(.12 <= nx.numeric_assortativity_coefficient(graph,'initial_stance') <= .20)

    def test_persona_only_changes_profile(self):
        persona=ec.Persona('A001','Example bio',10,-2)
        bullet=ec.build_prompt(persona,-2,['A002'],[],1,15)
        prose=ec.build_prompt(persona,-2,['A002'],[],1,15,'NARRATIVE_PROSE')
        self.assertEqual(bullet.split('Simulation rules:')[1],prose.split('Simulation rules:')[1])
        self.assertIn('strongly oppose stronger regulation',prose)
        self.assertIn('"target_index"',bullet)
        self.assertEqual(ec.clamp_stance(-2,2),-2)

    def test_provider_decoding(self):
        client=Mock()
        client.chat.completions.create.return_value.choices=[Mock(message=Mock(content='{}'))]
        for model in ('gpt-5.2','claude-haiku-4.5'):
            ec.call_model(client,model,'prompt',.3,.95)
            request=client.chat.completions.create.call_args.kwargs
            self.assertEqual('top_p' in request,not model.startswith('claude-'))
            self.assertEqual(request['temperature'],.3)


if __name__ == '__main__':
    unittest.main()
